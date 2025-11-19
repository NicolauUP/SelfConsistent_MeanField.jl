export MeanField_1stNeighbor_Repulsion, StartMeanField_1stNeighbor_Repulsion


function StartMeanField_1stNeighbor_Repulsion(InteractionData::NamedTuple, HamiltonianData::NamedTuple)
    U = InteractionData.U
    N_sites = HamiltonianData.N_sites
    H_MF = similar(HamiltonianData.H)
    fill!(H_MF, 0.0)

    type_of_guess = InteractionData.type_of_guess
    
    if type_of_guess == :CDW
        # Random initial guess for the mean-field Hamiltonian
        @inbounds for i in 1:N_sites      
            isodd(i) ? H_MF[i,i] = 1.0 : H_MF[i,i] = U * 0
        end

    else
        throw(ArgumentError("Unknown type_of_guess: $type_of_guess. Supported: :CDW"))
    end

    return (H_MF_int = H_MF,)
end      

function MeanField_1stNeighbor_Repulsion(ρ::AbstractMatrix, InteractionData::NamedTuple, HamiltonianData::NamedTuple)
    U = InteractionData.U
    N_sites = HamiltonianData.N_sites
    
    # Get local charge density at each site
    n_i = diag(ρ) |> real  
    c_dagi_c_ip1 = diag(ρ, 1)  #Upper diagonal \rho_i,i+1 =  ⟨c†_i+1 c_{i}⟩
    # Create H_MF with the same type and size as HamiltonianData.H
    H_MF = similar(HamiltonianData.H)

    #fill its entries all to 0.0
    fill!(H_MF, 0.0)
    
    @inbounds for i in 2:N_sites-1
        H_MF[i, i] = U *n_i[i-1]
        H_MF[i,i] += U * n_i[i+1]
    end

    @inbounds for i in 1:N_sites-1
        H_MF[i, i+1] = -U * c_dagi_c_ip1[i]
        H_MF[i+1, i] = -U * conj(c_dagi_c_ip1[i])
    end

    # Handle boundary sites for open boundary conditions
    if !HamiltonianData.periodic
        H_MF[1, 1] = U * n_i[2]
        H_MF[N_sites, N_sites] = U * n_i[N_sites - 1]
    else
        # Periodic boundary conditions
        H_MF[1, 1] = U * n_i[2] + U * n_i[N_sites]
        H_MF[N_sites, N_sites] = U * n_i[N_sites - 1] + U * n_i[1]

        H_MF[1, N_sites] = -U * ρ[1,N_sites]
        H_MF[N_sites, 1] = conj(H_MF[1, N_sites])
    end 

    return (H_MF_int = H_MF,)
end



