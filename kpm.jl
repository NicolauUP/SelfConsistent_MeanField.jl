using LinearAlgebra

export run_kpm_evolution, compute_jackson_kernel, compute_FermiDirac_coefs, chebyshev_polynomial, compute_lanczos_kernel


function chebyshev_polynomial(n::Integer, x::Real)
    return cos(n * acos(x))
end


function compute_jackson_kernel(kpm_params::NamedTuple) #Optimized!
    N_max = kpm_params.N_max
    J = zeros(Float64,N_max + 1)
    # handle trivial case to avoid division by zero (cot(pi) is singular)
    if N_max < 1
        J[1] = 1.0
        return J
    end

    den = N_max + 1
    θ0 = pi / den
    s,c = sincos(θ0)
    cot0 = c / s   # avoid relying on a possibly undefined cot()

    @inbounds for n in 0:N_max
        s, c = sincos(θ0 * n)
        J[n+1] = ((den - n) * c + s * cot0) / den
    end
    return J
end


function compute_lanczos_kernel(kpm_params::NamedTuple)
    N_max = kpm_params.N_max
    L = zeros(Float64, N_max + 1)
    L[1] = 1.0
    pi_over_Nmax_p1 = pi/(N_max + 1)
    @inbounds for n in 1:N_max
        constant = pi_over_Nmax_p1 * n
        L[n+1] = sin(constant) / constant
    end
    return L
end


function compute_FermiDirac_coefs(kpm_params::NamedTuple, E_Fermi::Real) #Zero temperature limit!!
    cs = zeros(Float64, kpm_params.N_max + 1)

    π_inv = 1 / π
    a = kpm_params.scaling_a
    b = kpm_params.scaling_b
    x_Fermi = (E_Fermi - b) / a

    if !isfinite(x_Fermi) || x_Fermi < -1.0 || x_Fermi > 1.0
        throw(DomainError(x_Fermi, "rescaled Fermi energy must be in [-1, 1] for acos; got $x_Fermi (a=$(a), b=$(b))"))
    end
   
    arccos_xF = acos(x_Fermi)


    cs[1] = 1 - arccos_xF * π_inv #The different case

    ns = 1:kpm_params.N_max
    cs[2:end] = @. -2 * sin(ns * arccos_xF) * π_inv / ns 


    return cs
end



function compute_spectral_coefs(kpm_params::NamedTuple)
    cs = zeros(Float64, kpm_params.N_max + 1, length(kpm_params.Energies))

    a = kpm_params.scaling_a
    b = kpm_params.scaling_b
    rescaled_energies = (kpm_params.Energies .- b) ./ a
    pi_inv = 1 / π
    sqrt_energies_inv = @. 1 / sqrt(1 - rescaled_energies^2)
    cs[1, :] = pi_inv .* sqrt_energies_inv



    ns = 1:kpm_params.N_max

    if !all(isfinite.(rescaled_energies)) || minimum(rescaled_energies) < -1.0 || maximum(rescaled_energies) > 1.0
        throw(DomainError(rescaled_energies, "rescaled energies must be in [-1, 1] for acos; got $rescaled_energies (a=$(a), b=$(b))"))
    end


    @inbounds for i_e in eachindex(rescaled_energies)
        arccos_e = acos(rescaled_energies[i_e])
        cs[2:end, i_e] = @. 2 * pi_inv * cos(ns * arccos_e) * sqrt_energies_inv[i_e]
    end
    return cs
end

function compute_charge_density(kpm_result::NamedTuple,coefs::Vector{Float64},HamiltonianData::NamedTuple)
    ρ = sum(coefs .* kpm_result.moments .* kpm_result.kernel_coefs)
    return ρ / HamiltonianData.N_sites
end

function find_new_FermiEnergy(target::Real,kpm_result::NamedTuple, HamiltonianData::NamedTuple, kpm_params::NamedTuple; tol=1e-6, max_iter=400, search_range=3.0,)
#Actual estimator of the gap would be interesting to allow for a dynamic search range!
    
    # initial search window (this +- 2 is arbitrary and may be adjusted by the caller)
    E_min = HamiltonianData.E_Fermi - search_range / 2
    E_max = HamiltonianData.E_Fermi + search_range / 2

    # ensure the rescaled interval intersects the valid domain [-1, 1] used by acos in compute_FermiDirac_coefs
    a = kpm_params.scaling_a
    b = kpm_params.scaling_b
    x_min = (E_min - b) / a
    x_max = (E_max - b) / a

    if !isfinite(x_min) || !isfinite(x_max)
        throw(DomainError((x_min, x_max), "Rescaled search interval contains non-finite values: $(x_min), $(x_max)"))
    end

    # if the entire interval lies outside [-1,1], we cannot evaluate acos reliably for any point in it
    if (x_min > 1.0 && x_max > 1.0) || (x_min < -1.0 && x_max < -1.0)
        throw(DomainError((x_min, x_max), "Rescaled search interval [$(x_min), $(x_max)] lies completely outside [-1, 1]; adjust search range or kpm_params"))
    end

    for _ in 1:max_iter
        E_mid = (E_min + E_max) / 2
        coefs = compute_FermiDirac_coefs(kpm_params, E_mid)

        ρ = compute_charge_density(kpm_result, coefs, HamiltonianData)

        if abs(ρ - target) < tol
            return E_mid
        elseif ρ < target
            E_min = E_mid
        else
            E_max = E_mid
        end
    end
    throw(ErrorException("Failed to converge to target charge density $target within $max_iter iterations"))
end


function chebyshev_step!(T_new::AbstractMatrix, 
    T_curr::AbstractMatrix,
    T_prev::AbstractMatrix,
    H_scaled::AbstractMatrix)  #notiche the ! -> this is an in-place operation.

    # 1. Calculate T_new = 2 H_scaaled * T_curr 
    # mul!(C, A, B, α, β) computes C = α*A*B + β*C so we set β=0.0 to overwrite T_new
    mul!(T_new, H_scaled, T_curr, 2.0, 0.0) # T_new = 2 H_scaled * T_curr

    # 2. Subtract T_prev from T_new in-place

    @. T_new -= T_prev 
    return nothing
end





function run_kpm_evolution(HamiltonianData::NamedTuple, kpm_params::NamedTuple)
    N_sites = HamiltonianData.N_sites

    # --- 1. Rescale Hamiltonian ---
    I_N = one(HamiltonianData.H)
    H_scaled = (HamiltonianData.H - kpm_params.scaling_b * I_N) / kpm_params.scaling_a 

    el_type = eltype(HamiltonianData.H)

    # --- 2. Create storage for Chebyshev polynomials, T_n(H) ---
    T_prev = zeros(el_type , N_sites, N_sites)
    T_curr = zeros(el_type , N_sites, N_sites)
    T_new = zeros(el_type , N_sites, N_sites)

    #  --- 3. Buffers to save moments and the density matrix ---
    moments = zeros(Float64, kpm_params.N_max + 1)
    
    ρ = nothing
    A = nothing  # Spectral function buffer
    LDOS = nothing
     if kpm_params.compute_spectral_function && isempty(kpm_params.Energies)
        throw(ArgumentError("'compute_local_density_of_states' is true, but the 'Energies' list is empty. Please provide at least one energy point."))
    end

    if kpm_params.compute_density_matrix 
        ρ = zeros(el_type, N_sites, N_sites)
        fermi_coefs = compute_FermiDirac_coefs(kpm_params, HamiltonianData.E_Fermi)
    end
    if kpm_params.compute_spectral_function
        # This is the massive (N, N, E) matrix
        A = zeros(el_type, N_sites, N_sites, length(kpm_params.Energies)) 
        spectral_coefs = compute_spectral_coefs(kpm_params)
    end
    if kpm_params.compute_ldos 
        LDOS = zeros(Float64, N_sites, length(kpm_params.Energies))
        spectral_coefs = compute_spectral_coefs(kpm_params)
    end
    if ρ === nothing && A === nothing && LDOS === nothing
        throw(ArgumentError("Either compute_density_matrix or compute_ldos or compute_spectral_function must be true in kpm_params"))
    end


    
    # --- 4. Compute kernel and Fermi-Dirac coefficients ---
    if kpm_params.kernel == :jackson
        kernel_coefs = compute_jackson_kernel(kpm_params)
    elseif kpm_params.kernel == :lanczos
        kernel_coefs = compute_lanczos_kernel(kpm_params)
    else
        throw(ArgumentError("Unknown kernel type: $(kpm_params.kernel). Supported types are :jackson and :lanczos"))
    end


    # --- 3. Initialize Chebyshev polynomials ---

    copyto!(T_prev, Matrix{el_type}(I, N_sites, N_sites))  # T_0(H) = I
    copyto!(T_curr, H_scaled)                                 # T_1(H) = H

    moments[1] = tr(T_prev) |> real  # μ_0 = Tr[T_0(H)] = Tr[I] = N_sites
    moments[2] = tr(T_curr) |> real  # μ_1 = Tr[T_1(H)] = Tr[H_scaled]
    
    if kpm_params.compute_density_matrix
        ρ .+= fermi_coefs[1] * kernel_coefs[1] * T_prev 
        ρ .+= fermi_coefs[2] * kernel_coefs[2] * T_curr
    end

    if kpm_params.compute_spectral_function
        for i_e in eachindex(kpm_params.Energies)

            A[:,:,i_e] .+= spectral_coefs[1,i_e] * kernel_coefs[1] * T_prev 
            A[:,:,i_e] .+= spectral_coefs[2,i_e] * kernel_coefs[2] * T_curr
        end
    end

    if kpm_params.compute_ldos
        for i_e in eachindex(kpm_params.Energies)
                LDOS[:, i_e] += spectral_coefs[1,i_e] * kernel_coefs[1] * diag(T_prev) |> real
                LDOS[:, i_e] += spectral_coefs[2,i_e] * kernel_coefs[2] * diag(T_curr) |> real
        end
    end
    
    # --- 4. Main KPM Loop ---
    @inbounds for n in 1:kpm_params.N_max-1
        # Compute T_{n+1}(H) using the recurrence relation
        chebyshev_step!(T_new, T_curr, T_prev, H_scaled)

        # Compute the moment μ_{n+1} = Tr[T_{n+1}(H)]
        moments[n+2] = tr(T_new) |> real
        if kpm_params.compute_density_matrix
            ρ .+= fermi_coefs[n+2] * kernel_coefs[n+2] * T_new
        end
        if kpm_params.compute_spectral_function
            for i_e in eachindex(kpm_params.Energies)
                A[:,:,i_e] .+= spectral_coefs[n+2,i_e] * kernel_coefs[n+2] * T_new
            end
        end

        if kpm_params.compute_ldos
            for i_e in eachindex(kpm_params.Energies)
                LDOS[:, i_e] += spectral_coefs[n+2,i_e] * kernel_coefs[n+2] * diag(T_new) |> real
            end
        end

        (T_prev, T_curr, T_new) = (T_curr, T_new, T_prev)  # Rotate references for next iteration
    end


    return (moments=moments, density_matrix=ρ, spectral_function=A,LDOS=LDOS, kernel_coefs=kernel_coefs, fermi_coefs=fermi_coefs,)
end



