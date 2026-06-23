#!/usr/bin/env julia

using Printf
using LinearAlgebra
using ForwardDiff

include(joinpath(@__DIR__, "train_a1_a2.jl"))

function parse_float_arg(key::AbstractString, default::Float64)
    for i in 1:length(ARGS)-1
        if ARGS[i] == key
            return parse(Float64, ARGS[i + 1])
        end
    end
    return default
end

function parse_int_arg(key::AbstractString, default::Int)
    for i in 1:length(ARGS)-1
        if ARGS[i] == key
            return parse(Int, ARGS[i + 1])
        end
    end
    return default
end

function tag_param(x::Real)
    s = @sprintf("%.6f", x)
    return replace(s, "-" => "m", "+" => "p", "." => "p")
end

function temperature_grid(T0_mev::Float64, Tmax_mev::Float64, step_mev::Float64)
    vals = Float64[]
    t = T0_mev
    while t <= Tmax_mev + 1e-10
        push!(vals, t)
        t += step_mev
    end
    if abs(vals[end] - Tmax_mev) > 1e-8
        push!(vals, Tmax_mev)
    end
    return vals
end

function solve_order_param(T, ints, x0::AbstractVector, a1p::Float64, a2p::Float64)
    mu_B = zero(T)
    fWrapper(Xs) = Quark_mu_param(Xs, mu_B, T, ints, a1p, a2p)
    res = nlsolve(fWrapper, copy(x0), autodiff = :forward, ftol = 1e-12, xtol = 1e-12, iterations = 600)
    x = res.zero
    return x, norm(fWrapper(x))
end

function solve_order_best(T, ints, candidates::Vector{Vector{Float64}}, a1p::Float64, a2p::Float64)
    best_x = copy(candidates[1])
    best_norm = Inf
    for cand in candidates
        try
            x, rnorm = solve_order_param(T, ints, cand, a1p, a2p)
            if all(isfinite, x) && isfinite(rnorm) && rnorm < best_norm
                best_x = x
                best_norm = rnorm
            end
        catch
        end
    end
    return best_x, best_norm
end

function cV_ad_implicit(X::AbstractVector, T::Float64, ints, a1p::Float64, a2p::Float64)
    mus = [0.0, 0.0, 0.0]
    z0 = vcat(Float64.(X), T)
    H = ForwardDiff.hessian(z -> Omega_param(z[1:5], mus, z[6], ints, a1p, a2p), z0)

    Hxx = Symmetric(Matrix{Float64}(H[1:5, 1:5]))
    HxT = Vector{Float64}(H[1:5, 6])
    HTT = Float64(H[6, 6])

    # Along the equilibrium branch Omega_X = 0:
    # P''(T) = -Omega_TT + Omega_TX * inv(Omega_XX) * Omega_XT.
    P_TT = -HTT + dot(HxT, Hxx \ HxT)
    return T * P_TT
end

function run_eos()
    a1p = parse_float_arg("--a1", -7.796309277317753)
    a2p = parse_float_arg("--a2", 0.11665983548204843)
    p_num = parse_int_arg("--p_num", 120)
    Tmin_mev = parse_float_arg("--Tmin", 50.0)
    Tmax_mev = parse_float_arg("--Tmax", 400.0)
    step_mev = parse_float_arg("--Tstep", 5.0)
    T0_mev = parse_float_arg("--T0", 10.0)

    if T0_mev >= Tmin_mev
        error("T0 must be below Tmin so that P0 can be used as the pressure baseline.")
    end

    ints = get_nodes(p_num)
    mus = [0.0, 0.0, 0.0]
    T_all_mev = temperature_grid(T0_mev, Tmax_mev, step_mev)
    T_all = T_all_mev ./ hc

    cold_base = [-1.88, -1.88, -2.60, 0.001, 0.001]
    candidates = Vector{Vector{Float64}}()
    push!(candidates, cold_base)
    push!(candidates, [-1.8, -1.8, -2.2, 0.1, 0.1])
    push!(candidates, [-1.9, -1.9, -2.7, 0.02, 0.02])
    push!(candidates, [-1.7, -1.7, -2.4, 0.05, 0.05])

    X0, r0 = solve_order_best(T_all[1], ints, candidates, a1p, a2p)
    omega0 = Float64(Omega_param(X0, mus, T_all[1], ints, a1p, a2p))

    n = length(T_all)
    omega = Vector{Float64}(undef, n)
    pressure = Vector{Float64}(undef, n)
    entropy = Vector{Float64}(undef, n)
    energy = Vector{Float64}(undef, n)
    trace = Vector{Float64}(undef, n)
    cV = Vector{Float64}(undef, n)
    residual = Vector{Float64}(undef, n)

    X = X0
    println("EOS at best-fit a1/a2")
    @printf("a1=%.12f, a2=%.12f, p_num=%d\n", a1p, a2p, p_num)
    @printf("pressure baseline: P0 from T=%.6f MeV, muB=0, residual=%.6e\n", T0_mev, r0)
    println("T_MeV,s/T^3,P/T^4,e/T^4,I/T^4,cV/T^3,cs2")

    for i in eachindex(T_all)
        if i > 1
            X, residual[i] = solve_order_param(T_all[i], ints, X, a1p, a2p)
        else
            residual[i] = r0
        end

        omega[i] = Float64(Omega_param(X, mus, T_all[i], ints, a1p, a2p))
        pressure[i] = -(omega[i] - omega0)
        entropy[i] = Float64(-dOmega_dT_param(X, mus, T_all[i], ints, a1p, a2p))
        energy[i] = -pressure[i] + T_all[i] * entropy[i]
        trace[i] = energy[i] - 3.0 * pressure[i]
        cV[i] = cV_ad_implicit(X, T_all[i], ints, a1p, a2p)
    end

    cs2 = entropy ./ cV

    out_path = joinpath(@__DIR__, @sprintf("eos_best_a1_%s_a2_%s.csv", tag_param(a1p), tag_param(a2p)))
    open(out_path, "w") do io
        println(io, "T_MeV,s_over_T3,P_over_T4,e_over_T4,I_over_T4,cV_over_T3,cs2")
        for i in eachindex(T_all)
            if T_all_mev[i] + 1e-8 < Tmin_mev
                continue
            end
            T = T_all[i]
            sT3 = entropy[i] / T^3
            PT4 = pressure[i] / T^4
            eT4 = energy[i] / T^4
            IT4 = trace[i] / T^4
            cVT3 = cV[i] / T^3
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                T_all_mev[i], sT3, PT4, eT4, IT4, cVT3, cs2[i])
            @printf("%.1f,% .8f,% .8f,% .8f,% .8f,% .8f,% .8f\n",
                T_all_mev[i], sT3, PT4, eT4, IT4, cVT3, cs2[i])
        end
    end

    @printf(">> saved EOS csv: %s\n", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_eos()
end
