

include("meanfield.jl")
include("kpm.jl")
include("models.jl")
include("types.jl")
include("solver_wrappers.jl")
using Printf # Import for formatted printing
using LinearAlgebra

function run_scf_loop(
    # --- Input Parameters ---
    HamiltonianData::NamedTuple,
    InteractionData::NamedTuple,
    scf_Params::NamedTuple,
    # --- Solver Configuration ---
    density_solver::AbstractDensitySolver,
    # --- Function Arguments
    hamiltonian_builder::Function,
    meanfield_starter::Function,
    meanfield_updater::Function
    ;
    Pulay_max_history::Int = 3)

    println("--- Starting SCF loop ---")

    println("Hamiltonian constructed from: $(hamiltonian_builder)")
    println("Mean-field starter: $(meanfield_starter)")
    println("Mean-field updater: $(meanfield_updater)")
    println("Density solver: $(density_solver)")
    # --- 1.5. Initialize buffers to store SCF loop data
    scf_loop_buffers = (
        errors = Float64[], 
        E_Fermi_values = Float64[],
        eigenvalues = [],#Only if ObtainEigenvalues = true
        solver_time = Float64[]
    )

    # --- 2.Get H0 
    if scf_Params.mixing == :Pulay
        buffer_errors = []
        buffer_H_MF = []
    end


    
    HamiltonianData = hamiltonian_builder(HamiltonianData)
    result = nothing
    density_matrix = nothing
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
        HamiltonianData_Current = (; HamiltonianData..., H = H_total)  #copies everything from HamiltonianData and replaces H with H_total


        #We assume that the first iteration has the correct Femi Energy!
        #It should be, we choose the initial guess accordingly!

        solver_time = @elapsed begin
            (density_matrix, E_Fermi, result) = obtain_density(density_solver, HamiltonianData_Current, InteractionData.density_target) #This will automatically call the correct function based on the type of density_solver (TOP) This could be easily generalized to MPOs!
        end

        if scf_Params.verbose == 2
            @printf("  -> Solver time: %.9f s\n", solver_time)
        end

        H_MF_int_new = meanfield_updater(density_matrix, InteractionData, HamiltonianData)


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


        #= This could be encapsualted in a function, but for now I leave it here for clarity =#

        if scf_Params.mixing == :Linear || scf_iter < Pulay_max_history 
        H_MF_int_current = (
            H_MF_int = scf_Params.mixing_parameter * H_MF_int_new.H_MF_int + 
                      (1 - scf_Params.mixing_parameter) * H_MF_int_current.H_MF_int,
        )
        end

        if scf_Params.mixing == :Pulay && scf_iter >= Pulay_max_history
            #Store current error and H_MF
            buffer_H_MF = push!(buffer_H_MF, H_MF_int_new.H_MF_int)
            buffer_errors = push!(buffer_errors, H_MF_int_new.H_MF_int - H_MF_int_current.H_MF_int) #difference between full new and old mean-field Hamiltonians (maybe using only diagonal would be better? and faster? cheaper in memory?)

            if length(buffer_H_MF) > Pulay_max_history
                popfirst!(buffer_H_MF)
                popfirst!(buffer_errors)
            end #Keep only the last Pulay_max_history elements

            #Build the Pulay mixing
            N_buf = length(buffer_H_MF)
            F_mat = zeros(Float64, N_buf + 1 , N_buf + 1)
            for i in 1:N_buf
                for j in 1:N_buf
                    F_mat[i,j] = real(sum(dot(buffer_errors[i] , buffer_errors[j])))
                end     
            end
            F_mat[end, 1:end-1] .= 1.0
            F_mat[1:end-1, end] .= -1.0

            b_vec = zeros(Float64, N_buf + 1)
            b_vec[end] = 1.0

            #=
            Introduce regularization to avoid singular matrix issues
            This adds a small value to the diagonal elements of F_mat
            which helps to stabilize the inversion process.
            =#

            F_mat += 1e-12 * I

            coeffs = F_mat \ b_vec  #Solve the linear system


            H_MF_int_current = (
                H_MF_int = sum( coeffs[i] * buffer_H_MF[i] for i in 1:N_buf ),)

        end

        # --- 4.6 Update Fermi Level ---
        if scf_Params.ObtainEigenvalues
            push!(scf_loop_buffers.eigenvalues, eigvals(Matrix(H_total)) |> real)
        end

        if scf_Params.verbose == 2
            println("SCF Error: $error")
            println()
        end
        push!(scf_loop_buffers.E_Fermi_values, E_Fermi)
        push!(scf_loop_buffers.solver_time, solver_time)
        # kpm_params = (; kpm_params..., scaling_a = E_Fermi) -> I have to think on this!
        HamiltonianData = (; HamiltonianData..., E_Fermi  = E_Fermi) #Update Fermi Energy basically!
    end




    return (density_matrix = density_matrix, scf_loop = scf_loop_buffers,result = result,)
end