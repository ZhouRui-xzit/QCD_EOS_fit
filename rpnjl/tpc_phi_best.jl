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

function range_values(lo::Float64, hi::Float64, step::Float64)
    vals = Float64[]
    t = lo
    while t <= hi + 1e-10
        push!(vals, t)
        t += step
    end
    if abs(vals[end] - hi) > 1e-8
        push!(vals, hi)
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

function dphi_dT_ad(X::AbstractVector, T::Float64, ints, a1p::Float64, a2p::Float64)
    mus = [0.0, 0.0, 0.0]
    z0 = vcat(Float64.(X), T)
    H = ForwardDiff.hessian(z -> Omega_param(z[1:5], mus, z[6], ints, a1p, a2p), z0)
    Hxx = Symmetric(Matrix{Float64}(H[1:5, 1:5]))
    HxT = Vector{Float64}(H[1:5, 6])

    # Equilibrium branch: Omega_X(X(T), T) = 0, so X_T = -Omega_XX^{-1} Omega_XT.
    dX_dT_model = -(Hxx \ HxT)
    return dX_dT_model ./ hc
end

function compute_curve(Ts_mev::Vector{Float64}, ints, a1p::Float64, a2p::Float64; x_start = nothing)
    cold_base = [-1.88, -1.88, -2.60, 0.001, 0.001]
    candidates = Vector{Vector{Float64}}()
    if x_start !== nothing
        push!(candidates, copy(x_start))
    end
    push!(candidates, cold_base)
    push!(candidates, [-1.8, -1.8, -2.2, 0.1, 0.1])
    push!(candidates, [-1.9, -1.9, -2.7, 0.02, 0.02])
    push!(candidates, [-1.7, -1.7, -2.4, 0.05, 0.05])

    n = length(Ts_mev)
    Xs = Vector{Vector{Float64}}(undef, n)
    dphis = Matrix{Float64}(undef, n, 3)
    residuals = Vector{Float64}(undef, n)

    X, rnorm = solve_order_best(Ts_mev[1] / hc, ints, candidates, a1p, a2p)
    for i in eachindex(Ts_mev)
        if i > 1
            X, rnorm = solve_order_param(Ts_mev[i] / hc, ints, X, a1p, a2p)
        end
        Xs[i] = Float64.(X)
        dphis[i, :] .= dphi_dT_ad(X, Ts_mev[i] / hc, ints, a1p, a2p)[1:3]
        residuals[i] = rnorm
    end

    return Xs, dphis, residuals
end

function max_row(Ts, Xs, dphis, idx::Int)
    k = argmax(dphis[:, idx])
    return (
        index = k,
        T = Ts[k],
        phi = Xs[k][idx],
        dphi = dphis[k, idx],
    )
end

function run_tpc()
    a1p = parse_float_arg("--a1", -7.796309277317753)
    a2p = parse_float_arg("--a2", 0.11665983548204843)
    p_num = parse_int_arg("--p_num", 120)
    Tmin = parse_float_arg("--Tmin", 50.0)
    Tmax = parse_float_arg("--Tmax", 400.0)
    coarse_step = parse_float_arg("--coarse_step", 1.0)
    fine_step = parse_float_arg("--fine_step", 0.05)
    fine_half_width = parse_float_arg("--fine_half_width", 4.0)

    ints = get_nodes(p_num)
    Ts = range_values(Tmin, Tmax, coarse_step)

    println("Tpc from max dphi/dT using implicit AD")
    @printf("a1=%.12f, a2=%.12f, p_num=%d\n", a1p, a2p, p_num)
    @printf("coarse scan: %.3f -> %.3f MeV, step=%.4f MeV\n", Tmin, Tmax, coarse_step)

    Xs, dphis, residuals = compute_curve(Ts, ints, a1p, a2p)
    coarse_u = max_row(Ts, Xs, dphis, 1)
    coarse_d = max_row(Ts, Xs, dphis, 2)
    coarse_s = max_row(Ts, Xs, dphis, 3)

    function fine_scan(center_T::Float64, flavor_idx::Int, label::AbstractString)
        lo = max(Tmin, center_T - fine_half_width)
        hi = min(Tmax, center_T + fine_half_width)
        fine_Ts = range_values(lo, hi, fine_step)
        start_idx = findlast(t -> t <= lo, Ts)
        start_x = start_idx === nothing ? Xs[1] : Xs[start_idx]
        @printf("fine scan around %s peak: %.3f -> %.3f MeV, step=%.4f MeV\n", label, lo, hi, fine_step)
        fine_Xs, fine_dphis, fine_residuals = compute_curve(fine_Ts, ints, a1p, a2p; x_start = start_x)
        return fine_Ts, fine_Xs, fine_dphis, fine_residuals, max_row(fine_Ts, fine_Xs, fine_dphis, flavor_idx)
    end

    fine_light_Ts, fine_light_Xs, fine_light_dphis, fine_light_residuals, fine_u =
        fine_scan(coarse_u.T, 1, "light")
    fine_d = max_row(fine_light_Ts, fine_light_Xs, fine_light_dphis, 2)
    fine_s_Ts, fine_s_Xs, fine_s_dphis, fine_s_residuals, fine_s =
        fine_scan(coarse_s.T, 3, "strange")

    println(">> coarse maxima")
    @printf("phi_u: Tpc=%.6f MeV, phi=%.12g, dphi/dT=%.12g\n", coarse_u.T, coarse_u.phi, coarse_u.dphi)
    @printf("phi_d: Tpc=%.6f MeV, phi=%.12g, dphi/dT=%.12g\n", coarse_d.T, coarse_d.phi, coarse_d.dphi)
    @printf("phi_s: Tpc=%.6f MeV, phi=%.12g, dphi/dT=%.12g\n", coarse_s.T, coarse_s.phi, coarse_s.dphi)

    println(">> fine maxima")
    @printf("phi_u: Tpc=%.6f MeV, phi=%.12g, dphi/dT=%.12g\n", fine_u.T, fine_u.phi, fine_u.dphi)
    @printf("phi_d: Tpc=%.6f MeV, phi=%.12g, dphi/dT=%.12g\n", fine_d.T, fine_d.phi, fine_d.dphi)
    @printf("phi_s: Tpc=%.6f MeV, phi=%.12g, dphi/dT=%.12g\n", fine_s.T, fine_s.phi, fine_s.dphi)

    out_path = joinpath(@__DIR__, @sprintf("tpc_phi_best_a1_%s_a2_%s.csv", tag_param(a1p), tag_param(a2p)))
    open(out_path, "w") do io
        println(io, "T_MeV,phi_u,phi_d,phi_s,dphi_u_dT_MeV,dphi_d_dT_MeV,dphi_s_dT_MeV,residual")
        for i in eachindex(Ts)
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                Ts[i], Xs[i][1], Xs[i][2], Xs[i][3], dphis[i, 1], dphis[i, 2], dphis[i, 3], residuals[i])
        end
    end

    fine_light_out_path = joinpath(@__DIR__, @sprintf("tpc_phi_best_fine_light_a1_%s_a2_%s.csv", tag_param(a1p), tag_param(a2p)))
    open(fine_light_out_path, "w") do io
        println(io, "T_MeV,phi_u,phi_d,phi_s,dphi_u_dT_MeV,dphi_d_dT_MeV,dphi_s_dT_MeV,residual")
        for i in eachindex(fine_light_Ts)
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                fine_light_Ts[i], fine_light_Xs[i][1], fine_light_Xs[i][2], fine_light_Xs[i][3],
                fine_light_dphis[i, 1], fine_light_dphis[i, 2], fine_light_dphis[i, 3], fine_light_residuals[i])
        end
    end

    fine_s_out_path = joinpath(@__DIR__, @sprintf("tpc_phi_best_fine_strange_a1_%s_a2_%s.csv", tag_param(a1p), tag_param(a2p)))
    open(fine_s_out_path, "w") do io
        println(io, "T_MeV,phi_u,phi_d,phi_s,dphi_u_dT_MeV,dphi_d_dT_MeV,dphi_s_dT_MeV,residual")
        for i in eachindex(fine_s_Ts)
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                fine_s_Ts[i], fine_s_Xs[i][1], fine_s_Xs[i][2], fine_s_Xs[i][3],
                fine_s_dphis[i, 1], fine_s_dphis[i, 2], fine_s_dphis[i, 3], fine_s_residuals[i])
        end
    end

    println(">> saved coarse curve: ", out_path)
    println(">> saved fine light curve: ", fine_light_out_path)
    println(">> saved fine strange curve: ", fine_s_out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tpc()
end
