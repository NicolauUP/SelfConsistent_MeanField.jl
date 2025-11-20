using LinearAlgebra



function run_evolution(HamiltonianData::NamedTuple, params::NamedTuple,density_target::Real)
    
    # Placeholder for KPM evolution code
    
    FilledStates = round(Int, density_target* HamiltonianData.N_sites)
    FilledStates = clamp(FilledStates, 0, HamiltonianData.N_sites)  # Ensure within valid range  

    Es, Vs = eigen(Matrix(HamiltonianData.H)) #Ensure that H is a Matrix
    if FilledStates == 0
        density_matrix = zeros(size(HamiltonianData.H))
        E_fermi = Es[1] - 1.0  # Below the lowest energy level
    
    elseif FilledStates == HamiltonianData.N_sites
        density_matrix = I(size(HamiltonianData.H, 1))
        E_fermi = Es[end] + 1.0  # Above the highest energy level
    else

        density_matrix = Vs[1:end, 1:FilledStates] * Vs[1:end, 1:FilledStates]'  # Construct density matrix from occupied states
        E_fermi = (Es[FilledStates] + Es[FilledStates + 1]) / 2  # Midpoint between highest occupied and lowest unoccupied energy levels (Being consistent with every method!)
    end
    other_data = (;)
    if get(params,:Save_Eigenvectors, false)
        other_data = (; other_data..., eigenvectors = Vs)    
    end
    if get(params,:Save_Eigenvalues, false)
        other_data = (other_data..., eigenvalues = Es)
    end
    return (density_matrix = density_matrix, E_Fermi = E_fermi, other_data = other_data)
end