

include("meanfield.jl")
include("kpm.jl")
include("models.jl")
using Printf # Import for formatted printing


function run_scf_loop(
    # --- Input Parameters ---
    HamiltonianData::NamedTuple,
    InteractionData::NamedTuple,
    kpm_params::NamedTuple,
    scf_Params::NamedTuple,

    # --- Function Arguments
    hamiltonian_builder::Function,
    meanfield_starter::Function,
    meanfield_updater::Function
)

    println("--- Starting SCF loop ---")

    println("Hamiltonian constructed from: $(hamiltonian_builder)")
    println("Mean-field starter: $(meanfield_starter)")
    println("Mean-field updater: $(meanfield_updater)")

    # --- 1.5. Initialize buffers to store SCF loop data
    scf_loop_buffers = (
        errors = Float64[],
        E_Fermi_values = Float64[],
        eigenvalues = [],
        kpm_time = [],
        fermi_time = []  #Only if ObtainEigenvalues = true
    )

    # --- 2.Get H0

    
    HamiltonianData = hamiltonian_builder(HamiltonianData)
    kpm_result = nothing
    # --- 3. Get initial mean-field Hamiltonian guess
    H_MF_int_current = meanfield_starter(InteractionData, HamiltonianData)
    diagonal_H_MF = diag(H_MF_int_current.H_MF_int) |> real
    
    # --- 4. Start SCF loop ---
    for scf_iter in 1:scf_Params.max_iterations
        if scf_Params.verbose == 1 || scf_Params.verbose == 2
           @printf("=== SCF Iteration %d ===\n", scf_iter)
        end

        # --- 4.1. Build total Hamiltonian ---
        H_total = HamiltonianData.H + H_MF_int_current.H_MF_int
        HamiltonianData_to_KPM = (; HamiltonianData..., H = H_total)  #copies everything from HamiltonianData and replaces H with H_total


        #We assume that the first iteration has the correct Femi Energy!
        #It should be, we choose the initial guess accordingly!


        kpm_time = @elapsed begin 
            kpm_result = run_kpm_evolution(
            HamiltonianData_to_KPM,
            kpm_params
        )   

        end

        if scf_Params.verbose == 2
            @printf("  -> KPM evolution time: %.9f s\n", kpm_time)
        end

        H_MF_int_new = meanfield_updater(kpm_result, InteractionData, HamiltonianData)


        diagonal_H_MF_new = diag(H_MF_int_new.H_MF_int) |> real       

        # --- 4.4. Check convergence ---
        error = maximum(abs.(diagonal_H_MF_new - diagonal_H_MF))
        push!(scf_loop_buffers.errors, error)
        
        diagonal_H_MF = diag(H_MF_int_new.H_MF_int) |> real #Update for next iteration!


        if error < scf_Params.convergence_tol && (scf_Params.verbose == 1 || scf_Params.verbose == 2)
            println("SCF converged in $scf_iter iterations with error $error")
            break
        end

        # --- 4.5 Mixing and continue
        H_MF_int_current = (
            H_MF_int = scf_Params.mixing_parameter * H_MF_int_new.H_MF_int + 
                      (1 - scf_Params.mixing_parameter) * H_MF_int_current.H_MF_int,
        )

        # --- 4.6 Update Fermi Level ---
        if scf_Params.ObtainEigenvalues
            push!(scf_loop_buffers.eigenvalues, eigvals(Matrix(H_total)) |> real)
        end

        Fermi_time = @elapsed begin
            E_Fermi = find_new_FermiEnergy(InteractionData.density_target, kpm_result, HamiltonianData, kpm_params)
        end

        if scf_Params.verbose == 2
            @printf("  -> Fermi energy update time: %.9f s\n", Fermi_time)
            println("SCF Error: $error")
            println()
        end
        push!(scf_loop_buffers.E_Fermi_values, E_Fermi)
        push!(scf_loop_buffers.kpm_time, kpm_time)
        push!(scf_loop_buffers.fermi_time, Fermi_time)
        # kpm_params = (; kpm_params..., scaling_a = E_Fermi) -> I have to think on this!
        HamiltonianData = (; HamiltonianData..., E_Fermi  = E_Fermi) #Update Fermi Energy basically!
    end




    return (kpm_result = kpm_result, scf_loop_buffers = scf_loop_buffers)
end