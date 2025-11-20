


include("types.jl")
include("kpm.jl")
include("ed.jl")
export KPM_Solver, ED_Solver


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


