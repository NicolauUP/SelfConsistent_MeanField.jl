KPM-SCF in Julia

This repository contains a Julia implementation of a Self-Consistent Field (SCF) mean-field calculation for 1D interacting fermion models.

The core of the method avoids direct diagonalization (which scales as $O(N^3)$) and instead uses the Kernel Polynomial Method (KPM) to efficiently compute the density matrix (scaling as $O(N)$ or $O(N^2)$ depending on sparsity).

This code is being developed for PhD research.

Project Structure

All main logic resides in the /src directory:

src/models.jl: Defines the non-interacting Hamiltonian models (e.g., 1D chain, Aubry-Andre) as sparse matrices.

src/kpm.jl: Implements the KPM expansion, kernels (Jackson, Lanczos), and the calculation of the density matrix and Fermi-Dirac coefficients.

src/meanfield.jl: Implements the Hartree-Fock mean-field approximation for a first-neighbor repulsion ($U n_i n_{i+1}$).

src/run_scf.jl: Contains the main run_scf_loop that manages the self-consistent convergence.

Dependencies

This project requires Julia (v1.6 or higher).

The code relies on the following standard libraries:

LinearAlgebra

SparseArrays

How to Run (Driver Example)

To run a simulation, create a "driver" file (e.g., main.jl) in the root directory. This file is responsible for defining all parameters and calling the run_scf_loop.

Example main.jl:

using SparseArrays
using LinearAlgebra

# Include the source files
include("src/models.jl")
include("src/kpm.jl")
include("src/meanfield.jl")
include("src/run_scf.jl")

println("--- Preparing SCF simulation ---")

# --- 1. Define Hamiltonian Parameters (H0) ---
HamiltonianData = (
    N_sites = 100,
    t = 1.0,
    periodic = true,
    E_Fermi = 0.0, # Initial guess (for 1st iteration)

    # Example for one_dimensional_AubryAndre:
    # V = 1.5,
    # τ = (sqrt(5)-1)/2,
    # φ = 0.0,
)

# --- 2. Define Interaction Parameters (Hint) ---
InteractionData = (
    U = 2.0,                  # Interaction strength
    density_target = 0.5,     # Target density (e.g., 0.5 for half-filling)
    type_of_guess = :CDW      # Initial guess type (:CDW)
)

# --- 3. Define KPM Parameters ---
# It is CRITICAL to estimate the total bandwidth (H0 + H_MF)
E_min_est = -5.0
E_max_est = 5.0
scaling_a = (E_max_est - E_min_est) / 2.0  # (E_max - E_min) / 2
scaling_b = (E_max_est + E_min_est) / 2.0  # (E_max + E_min) / 2

kpm_params = (
    N_max = 1000,              # Number of moments (precision)
    scaling_a = scaling_a,     # Scaling parameter 'a'
    scaling_b = scaling_b,     # Scaling parameter 'b'
    kernel = :jackson,         # KPM kernel (:jackson or :lanczos)

    # What to compute
    compute_density_matrix = true, # Mandatory for SCF
    compute_spectral_function = false,
    compute_ldos = false,
    Energies = []
)

# --- 4. Define SCF Loop Parameters ---
scf_Params = (
    max_iterations = 200,
    convergence_threshold = 1e-6,
    mixing_parameter = 0.1 # Mixing parameter (low and stable)
)

# --- 5. Define Functions and Execute ---

# Choose the H0 model
hamiltonian_builder = one_dimensional_chain 
# hamiltonian_builder = one_dimensional_AubryAndre

# Define the mean-field functions
meanfield_starter = StartMeanField_1stNeighbor_Repulsion
meanfield_updater = MeanField_1stNeighbor_Repulsion

println("--- Starting SCF loop ---")

(final_kpm_result, scf_history) = run_scf_loop(
    HamiltonianData,
    InteractionData,
    kpm_params,
    scf_Params,

    hamiltonian_builder,
    meanfield_starter,
    meanfield_updater
)

# --- 6. Analyze Results ---
println("--- Run Finished ---")
println("Error history: ", scf_history.errors)
println("Final E_Fermi: ", scf_history.E_Fermi_values[end])

# Final charge density profile
final_density_profile = diag(final_kpm_result.density_matrix) |> real
println("Final density profile:")
println(final_density_profile)


Next Steps (To-Do)

[ ] Implement a more advanced mixing method (e.g., Pulay/DIIS) to accelerate convergence.

[ ] Implement adaptive linear mixing.

[ ] Add calculation of the total system energy at the end of the SCF.

[ ] Add more models and interaction types (e.g., onsite Hubbard).