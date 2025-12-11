


include("types.jl")
include("kpm.jl")
include("ed.jl")
include("purification.jl")
export obtain_density


function obtain_density(solver::KPM_Solver, HamiltonianData::NamedTuple, density_target::Real)
    kpm_params = solver.params
    kpm_result = run_kpm_evolution(HamiltonianData, kpm_params)

    E_fermi = find_new_FermiEnergy(density_target,kpm_result, HamiltonianData, kpm_params)

    raw_result = (
        moments = kpm_result.moments,
        spectral_function = kpm_result.spectral_function,
        LDOS = kpm_result.LDOS,
        kernel_coefs = kpm_result.kernel_coefs,
        fermi_coefs = kpm_result.fermi_coefs
    )

    #= this raw result is a way to return all the KPM data ensuring no duplication of memory! this new structure just references the data already in kpm_result! 

    =#
    return (kpm_result.density_matrix, E_fermi, raw_result) #Return density matrix, Fermi energy and full kpm_result
end


function obtain_density(solver::ED_Solver, HamiltonianData::NamedTuple, density_target::Real)
    ed_params = solver.params
    ed_result = run_evolution(HamiltonianData, ed_params, density_target)
    return (ed_result.density_matrix, ed_result.E_Fermi, ed_result.other_data) #Return density matrix, Fermi energy and full ed_result
end


function obtain_density(solver::Purification_Solver, HamiltonianData::NamedTuple, density_target::Real)
  purification_params = solver.params
    verbose = get(purification_params, :verbose, false)
    max_iters = get(purification_params, :max_iters, 60)

    # 1. Force conversion to Dense Matrix (like ED_Solver)
    # This aligns with the benchmark requirement: "purification deals with dense matrix-matrix multiplications" 
    H_dense = Matrix(HamiltonianData.H)

    N_sites = HamiltonianData.N_sites
    Ne = round(Int, density_target * N_sites)

    # 2. Estimate Spectral Bounds (Needed for scaling [cite: 52])
    # Since we are already working with dense matrices, we can use eigvals efficiently for these sizes.


    H_min = get(HamiltonianData, :H_min, -5.0)
    H_max = get(HamiltonianData, :H_max, 5.0) 


    # 3. Run Purification
    # Note: We pass Ne (integer electrons) as required by Canonical Purification [cite: 66]
    ρ, run_stats = perform_purification(H_dense, Ne, H_min, H_max; MAX_ITER=max_iters, verbose=verbose)

    # 4. Construct Result Tuple
    # Purification yields the Total Energy directly[cite: 105], but not the Fermi Energy.
    # We keep the previous E_Fermi to preserve type stability in the SCF loop.
    E_Fermi = HamiltonianData.E_Fermi 

    

    raw_result = (
        density_errors = run_stats.density_errors,
        energies = run_stats.energies,
        idempotency_errors = run_stats.idempotency_errors,
        charge_error = run_stats.charge_error
    )
    return (ρ, E_Fermi, raw_result)
end