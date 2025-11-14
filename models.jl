

export one_dimensional_chain, one_dimensional_chain_gapped, one_dimensional_AubryAndre, one_dimensional_ExtendedAubryAndre

function one_dimensional_chain(HamiltonianData::NamedTuple)
    N_sites = HamiltonianData.N_sites
    t = HamiltonianData.t


  
    I = Int[]
    J = Int[]
    Vals = Float64[]

    # nearest-neighbor hoppings
    for i in 1:(N_sites - 1)
        push!(I, i);     push!(J, i + 1); push!(Vals, -t)
        push!(I, i + 1); push!(J, i);     push!(Vals, -t)
    end

    if HamiltonianData.periodic && N_sites > 1
        push!(I, 1);       push!(J, N_sites); push!(Vals, -t)
        push!(I, N_sites); push!(J, 1);       push!(Vals, -t)
    end
    
    H = SparseArrays.sparse(I, J, Vals, N_sites, N_sites)
    return (; HamiltonianData..., H = H)
end

function one_dimensional_chain_gapped(HamiltonianData::NamedTuple)
    N_sites = HamiltonianData.N_sites
    t = HamiltonianData.t
    Δ = HamiltonianData.Δ

    I = Int[]
    J = Int[]
    Vals = Float64[]

    # nearest-neighbor hoppings
    for i in 1:(N_sites - 1)
        push!(I, i);     push!(J, i + 1); push!(Vals, -t)
        push!(I, i + 1); push!(J, i);     push!(Vals, -t)
    end

    if HamiltonianData.periodic && N_sites > 1
        push!(I, 1);       push!(J, N_sites); push!(Vals, -t)
        push!(I, N_sites); push!(J, 1);       push!(Vals, -t)
    end

    # staggered onsite potential
    for i in 1:N_sites
        val = isodd(i) ? Δ / 2 : -Δ / 2
        push!(I, i); push!(J, i); push!(Vals, val)
    end

    H = SparseArrays.sparse(I, J, Vals, N_sites, N_sites)
    return (; HamiltonianData..., H = H)
end

function one_dimensional_AubryAndre(HamiltonianData::NamedTuple)
    N_sites = HamiltonianData.N_sites
    t = HamiltonianData.t
    V = HamiltonianData.V
    τ = HamiltonianData.τ
    φ = HamiltonianData.φ
    
    I = Int[]
    J = Int[]
    Vals = Float64[]

    #N earest-neighbor hoppings
    for i in 1:(N_sites - 1)
        push!(I, i);     push!(J, i + 1); push!(Vals, -t)
        push!(I, i + 1); push!(J, i);     push!(Vals, -t)
    end

    if HamiltonianData.periodic && N_sites > 1
        push!(I, 1);       push!(J, N_sites); push!(Vals, -t)
        push!(I, N_sites); push!(J, 1);       push!(Vals, -t)
    end

    # staggered onsite potential
    for i in 1:N_sites
        val = V * cos(2pi * τ * (i-1) + φ)
        push!(I, i); push!(J, i); push!(Vals, val)
    end
    H = SparseArrays.sparse(I, J, Vals, N_sites, N_sites)

    return (; HamiltonianData..., H = H)
end
    

function one_dimensional_ExtendedAubryAndre(HamiltonianData::NamedTuple)
    N_sites = HamiltonianData.N_sites
    t = HamiltonianData.t
    V1 = HamiltonianData.V1
    τ_AA = HamiltonianData.τ_AA
    V2 = HamiltonianData.V2
    τ_2 = HamiltonianData.τ_2
    φ = HamiltonianData.φ

    I = Int[]
    J = Int[]
    Vals = Float64[]
    #N earest-neighbor hoppings
    for i in 1:(N_sites - 1)
        push!(I, i);     push!(J, i + 1); push!(Vals, -t - V2 * cos(2pi * τ_2 * (i-0.5) + φ))
        push!(I, i + 1); push!(J, i);     push!(Vals, -t - V2 * cos(2pi * τ_2 * (i-0.5) + φ))
    end

    if HamiltonianData.periodic && N_sites > 1
        push!(I, 1);       push!(J, N_sites); push!(Vals, -t - V2 * cos(2pi * τ_2 * (N_sites-0.5) + φ))
        push!(I, N_sites); push!(J, 1);       push!(Vals, -t - V2 * cos(2pi * τ_2 * (N_sites-0.5) + φ))
    end

    # staggered onsite potential
    for i in 1:N_sites
        val = V1 * cos(2pi * τ_AA * (i-1) + φ)
        push!(I, i); push!(J, i); push!(Vals, val)
    end
    H = SparseArrays.sparse(I, J, Vals, N_sites, N_sites)

    return (; HamiltonianData..., H = H)
end