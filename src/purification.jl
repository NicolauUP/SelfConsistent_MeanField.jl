
using Printf

export perform_purification

"""
Constructs the initial density matrix ρ0 for Canonical Purification.
Implements Equations (11-14) from the provided notes.
"""
function construct_rho_0(N::Int, H::Matrix{Float64}, H_max::Float64, H_min::Float64, Ne::Int)
    μ = tr(H) / N
    λ = minimum((Ne /(H_max - μ), (N - Ne)/(μ - H_min)))

    rho_0 = similar(H)
    rho_0 = λ/N * (μ*I - H) + Ne/N * I
    return rho_0
end
function perform_purification(H::Matrix, Ne::Int, H_min::Float64, H_max::Float64; MAX_ITER::Int=60, verbose=true)
    N = size(H,1)

    # 1. Initialize ρ0
    ρ = construct_rho_0(N, H, H_max, H_min, Ne)

    # 2. Pre-allocate Buffers (Reusable Memory) 
    P2 = similar(ρ)
    P3 = similar(ρ)

    P2_temp = similar(ρ)

    
    charge_error_buffers = Float64[]
    idempotency_error_buffers = Float64[]
    energies = Float64[]
    density_error_buffers = Float64[]


    for i in 1:MAX_ITER
        mul!(P2, ρ, ρ)  # P2 = ρ0^2
        mul!(P3, ρ, P2)  # P3 =


        T1 = tr(ρ)
        T2 = tr(P2)
        T3 = tr(P3)

        denom = T1 - T2
    
        # Safey to avoid division by zero
        if abs(denom) < 1e-14
            if verbose println("Denominator too small, stopping iteration.") end
            break
        end

        cn = (T2 - T3) / denom

        c_ρ, c_p2, c_p3 = 0.0, 0.0 ,0.0

        if cn < 0.5
            inv_1_minus_cn = 1.0 / (1.0 - cn)
            c_ρ = (1 - 2*cn) * inv_1_minus_cn
            c_p2 = (1.0 + cn) * inv_1_minus_cn
            c_p3 = -1.0 * inv_1_minus_cn
        else
            inv_cn = 1.0 / cn
            c_ρ = 0.0
            c_p2 = (1.0 + cn) * inv_cn
            c_p3 = -1.0 * inv_cn
        end
        ρ_current = copy(ρ)

        # Fused In-Place Update
        @. ρ = c_ρ * ρ + c_p2 * P2 + c_p3 * P3
        



        current_trace = tr(ρ)
        err_charge = abs(current_trace - Ne) / Ne
        mul!(P2_temp, ρ, ρ)
        err_Idempotency = norm(P2_temp - ρ) / norm(ρ)



        if verbose 
            @printf("Iter %d: Charge Error = %.2e, Idem Error = %.2e\n", i, err_charge,err_Idempotency)
        end

        err_density = norm(ρ - ρ_current) / norm(ρ_current)
        push!(charge_error_buffers, err_charge)
        push!(energies, tr(H*ρ))
        push!(idempotency_error_buffers, err_Idempotency)
        push!(density_error_buffers, err_density)



        if (err_charge < 1e-7 && err_Idempotency < 1e-10) || err_density < 1e-10
            println("Converged at iteration $i")
            break
        end
    end
    return ρ, (density_errors=density_error_buffers, energies = energies, idempotency_errors = idempotency_error_buffers,charge_error = charge_error_buffers)
end
