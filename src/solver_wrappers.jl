


include("types.jl")
export KPM_Solver, ED_Solver


function obtain_density(solver::KPM_Solver, HamiltonianData::NamedTuple)
    kpm_params = solver.params
    kpm_result = run_kpm_evolution(HamiltonianData, kpm_params)
    return (kpm_result.sdensity_matrix, E_fermi, kpm_result) #Return density matrix, Fermi energy and full kpm_result
end


function obtain_density(solver::ED_Solver, HamiltonianData::NamedTuple)
    ed_params = solver.params
    ed_result = run_ed_solver(HamiltonianData, ed_params)
    return (ed_result.density_matrix, ed_result.E_Fermi, ed_result) #Return density matrix, Fermi energy and full ed_result
end


