include("run_scf.jl")
using ArgParse

function parse_command_line()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--algo"
            help = "The algorithm option - Purif, ED, KPM)"
            required = true
        "--N_max"
            help = "KPM expansion order"
            arg_type = Int
            default = 0
        "--N_sites"
            help = "Number of lattice sites"
            arg_type = Int
            required = true

        "--U"
            help = "Interaction strength"
            arg_type = Float64
            required = true
    end
    return parse_args(s)
end


args = parse_command_line()


scf_params = (
     max_iterations = 100,
     convergence_tol = 1e-5,
     mixing_parameter = 1.0,
     ObtainEigenvalues = false,
     verbose = 0,
     mixing=:Pulay)


HamiltonianData = (N_sites = args["N_sites"], t = 1.0, periodic=true, E_Fermi = 0.0)

# --- 2. Interaction Parameters ---

InteractionData = (U = args["U"],
                   type_of_guess=:CDW,
                   density_target = 0.5)


# --- 3. Solver Parameter ---
algo = args["algo"]
if algo == "Purif"
    purification_params = (max_iters = 100, verbose=false)
    @global solver = Purification_Solver(purification_params)

elseif algo == "ED"
    @global solver = ED_Solver(())

elseif algo == "KPM"
    N_max = args["N_max"]   
    ompute_density_matrix = true
    compute_ldos = false
    Energies = Float64[]
    compute_spectral_function = false 
    E_min = -5.0
    E_max = 5.0
    scaling_a = (E_max - E_min) / 2 * 1.1
    scaling_b = (E_max + E_min) / 2
    kpm_params = (
         N_max = N_max,
         compute_density_matrix = compute_density_matrix,
         compute_ldos = compute_ldos,
         compute_spectral_function = compute_spectral_function,
         Energies = Energies,
         kernel = :jackson,
         scaling_a = scaling_a,
         scaling_b = scaling_b)
    @global solver = KPM_Solver(kpm_params)


else
    error("Unknown algorithm: $(algo). Choose from Purif, ED, KPM")
end



# --- 5. Define the functions to be used ---
hamiltonian_builder = one_dimensional_chain
meanfield_starter = StartMeanField_1stNeighbor_Repulsion
meanfield_updater = MeanField_1stNeighbor_Repulsion

# --- 6. Execute the SCF loop ---
time_scf = @elapsed (density_matrix, scf_history, other_data) = run_scf_loop(
    HamiltonianData,
    InteractionData,
    scf_params,
     solver,

    hamiltonian_builder,
    meanfield_starter,
    meanfield_updater
;
Pulay_max_history = 4)

println("Elapsed time: ", time, " seconds")

