#!/usr/bin/env julia

using Printf
using LinearAlgebra
using ForwardDiff
using Optimization
using OptimizationOptimJL: NelderMead
using Plots

include(joinpath(@__DIR__, "rpnjl.jl"))

struct LogUFitData
    Ts::Vector{Float64}
    target::Vector{Float64}
    err::Vector{Float64}
    ints::Any
    x0_candidates::Vector{Vector{Float64}}
    theta_lo::Vector{Float64}
    theta_hi::Vector{Float64}
    tpc_target::Float64
    tpc_sigma::Float64
    tpc_grid::Vector{Float64}
end

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
    if isempty(vals) || abs(vals[end] - hi) > 1e-8
        push!(vals, hi)
    end
    return vals
end

sigmoid_stable(x) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))

function logit_clamped(x::Float64)
    xc = clamp(x, 1e-10, 1 - 1e-10)
    return log(xc / (1 - xc))
end

function theta_to_u(theta::AbstractVector, data::LogUFitData)
    return [logit_clamped((theta[i] - data.theta_lo[i]) / (data.theta_hi[i] - data.theta_lo[i])) for i in eachindex(theta)]
end

function u_to_theta(u::AbstractVector, data::LogUFitData)
    theta = Vector{Float64}(undef, length(u))
    for i in eachindex(u)
        s = sigmoid_stable(u[i])
        theta[i] = data.theta_lo[i] + (data.theta_hi[i] - data.theta_lo[i]) * s
    end
    return theta
end

function default_sT3_path()
    direct = joinpath(@__DIR__, "data", "sT3.csv")
    fallback = joinpath(@__DIR__, "sT3.csv")
    return isfile(direct) ? direct : fallback
end

function build_fit_data(;
    csv_path::String,
    p_num::Int,
    theta_lo::Vector{Float64},
    theta_hi::Vector{Float64},
    tpc_target::Float64,
    tpc_sigma::Float64,
    tpc_min::Float64,
    tpc_max::Float64,
    tpc_step::Float64,
)
    Ts, target, err = load_sT3_csv(csv_path)
    x0_candidates = [
        [-1.88, -1.88, -2.60, 0.001, 0.001],
        [-1.8, -1.8, -2.2, 0.1, 0.1],
        [-1.9, -1.9, -2.7, 0.02, 0.02],
        [-1.7, -1.7, -2.4, 0.05, 0.05],
    ]
    return LogUFitData(
        Ts,
        target,
        err,
        get_nodes(p_num),
        x0_candidates,
        theta_lo,
        theta_hi,
        tpc_target,
        tpc_sigma,
        range_values(tpc_min, tpc_max, tpc_step),
    )
end

function haar_argument(Phi1, Phi2)
    return 1 - 6 * Phi1 * Phi2 + 4 * (Phi1^3 + Phi2^3) - 3 * (Phi1 * Phi2)^2
end

function calc_U_log_fit(T, Phi1, Phi2, theta::AbstractVector)
    a1p, a2p, T0_mev = theta
    T0p = T0_mev / hc
    x = T0p / T
    aT = a0 + a1p * x * exp(-a2p * T / T0p)
    h = haar_argument(Phi1, Phi2)
    return T^4 * (-aT / 2 * Phi1 * Phi2 + b3 * x^3 * log(h))
end

function Omega_log_fit(orders, mus, T, ints, theta::AbstractVector)
    p1, w1 = ints[1]
    p2, w2 = ints[2]

    phi = orders[1:3]
    Phi1 = orders[4]
    Phi2 = orders[5]
    omega_total = chiral(phi) + calc_U_log_fit(T, Phi1, Phi2, theta)
    masses = Mass(phi)

    for flavor in 1:3
        mass = masses[flavor]
        mu = mus[flavor]
        omega_total += calculate_vacuum_term(p1, w1, mass)
        omega_total += calculate_thermal_term(p2, w2, mass, T, mu, Phi1, Phi2)
    end
    return omega_total
end

function dOmega_dorder_log_fit(orders, mus, T, ints, theta::AbstractVector)
    return ForwardDiff.gradient(x -> Omega_log_fit(x, mus, T, ints, theta), orders)
end

function dOmega_dT_log_fit(orders, mus, T, ints, theta::AbstractVector)
    return ForwardDiff.derivative(x -> Omega_log_fit(orders, mus, x, ints, theta), T)
end

function Quark_mu_log_fit(X0, mu_B, T, ints, theta::AbstractVector)
    Tout = promote_type(eltype(X0), typeof(T), typeof(mu_B), eltype(theta))
    orders = X0[1:5]
    mus = [mu_B / 3, mu_B / 3, mu_B / 3]
    fvec = zeros(Tout, 5)
    fvec[1:5] = dOmega_dorder_log_fit(orders, mus, T, ints, theta)
    return fvec
end

function solve_order_log_fit(T, ints, x0::AbstractVector, theta::AbstractVector)
    fWrapper(Xs) = Quark_mu_log_fit(Xs, zero(T), T, ints, theta)
    x = nonlinear_zero(fWrapper, x0; ftol = 1e-11, xtol = 1e-11, iterations = 500)
    rnorm = norm(fWrapper(x))
    h = haar_argument(x[4], x[5])
    if !all(isfinite, x) || !isfinite(rnorm) || !(h > 0)
        error("invalid order-parameter solution")
    end
    return Float64.(x), rnorm
end

function solve_order_best_log(T, data::LogUFitData, theta::AbstractVector)
    best_x = copy(data.x0_candidates[1])
    best_norm = Inf
    for cand in data.x0_candidates
        try
            x, rnorm = solve_order_log_fit(T, data.ints, cand, theta)
            if rnorm < best_norm
                best_x = x
                best_norm = rnorm
            end
        catch
        end
    end
    if !isfinite(best_norm)
        error("all initial guesses failed")
    end
    return best_x, best_norm
end

function dphi_dT_implicit_log(X::AbstractVector, T::Float64, ints, theta::AbstractVector)
    mus = [0.0, 0.0, 0.0]
    z0 = vcat(Float64.(X), T)
    H = ForwardDiff.hessian(z -> Omega_log_fit(z[1:5], mus, z[6], ints, theta), z0)
    Hxx = Symmetric(Matrix{Float64}(H[1:5, 1:5]))
    HxT = Vector{Float64}(H[1:5, 6])
    dX_dT_model = -(Hxx \ HxT)
    return dX_dT_model ./ hc
end

function parabolic_peak(xm::Float64, x0::Float64, xp::Float64, ym::Float64, y0::Float64, yp::Float64)
    denom = ym - 2 * y0 + yp
    if abs(denom) < 1e-14
        return x0, y0
    end
    step = xp - x0
    offset = 0.5 * step * (ym - yp) / denom
    if abs(offset) > step
        return x0, y0
    end
    ypeak = y0 - 0.25 * (ym - yp) * offset / step
    return x0 + offset, ypeak
end

function compute_predictions(data::LogUFitData, theta::AbstractVector)
    n = length(data.Ts)
    pred = Vector{Float64}(undef, n)
    residuals = Vector{Float64}(undef, n)
    phis = Matrix{Float64}(undef, n, 5)
    min_sT3 = Inf
    max_phi_violation = 0.0

    X, rnorm = solve_order_best_log(data.Ts[1] / hc, data, theta)
    for i in eachindex(data.Ts)
        if i > 1
            X, rnorm = solve_order_log_fit(data.Ts[i] / hc, data.ints, X, theta)
        end
        T = data.Ts[i] / hc
        s = Float64(-dOmega_dT_log_fit(X, [0.0, 0.0, 0.0], T, data.ints, theta))
        pred[i] = s / T^3
        residuals[i] = rnorm
        phis[i, :] .= X
        min_sT3 = min(min_sT3, pred[i])
        max_phi_violation = max(max_phi_violation, max(0.0, -X[4], -X[5], X[4] - 1, X[5] - 1))
    end
    return pred, residuals, phis, min_sT3, max_phi_violation
end

function compute_light_tpc(data::LogUFitData, theta::AbstractVector)
    n = length(data.tpc_grid)
    dlight = Vector{Float64}(undef, n)
    phis = Matrix{Float64}(undef, n, 5)

    X, _ = solve_order_best_log(data.tpc_grid[1] / hc, data, theta)
    for i in eachindex(data.tpc_grid)
        if i > 1
            X, _ = solve_order_log_fit(data.tpc_grid[i] / hc, data.ints, X, theta)
        end
        dX = dphi_dT_implicit_log(X, data.tpc_grid[i] / hc, data.ints, theta)
        dlight[i] = 0.5 * (dX[1] + dX[2])
        phis[i, :] .= X
    end

    k = argmax(dlight)
    tpc = data.tpc_grid[k]
    peak = dlight[k]
    if 1 < k < n
        tpc, peak = parabolic_peak(
            data.tpc_grid[k - 1],
            data.tpc_grid[k],
            data.tpc_grid[k + 1],
            dlight[k - 1],
            dlight[k],
            dlight[k + 1],
        )
    end
    return tpc, peak, dlight, phis
end

function evaluate_theta(theta::AbstractVector, data::LogUFitData; save_details::Bool = false)
    if any(theta .<= data.theta_lo) || any(theta .>= data.theta_hi)
        return (loss = 1e30, s_loss = 1e30, tpc_loss = 1e30, tpc = NaN, pred = Float64[], residuals = Float64[], phis = zeros(0, 0))
    end

    try
        pred, residuals, phis, min_sT3, max_phi_violation = compute_predictions(data, theta)
        s_loss = 0.0
        for i in eachindex(data.Ts)
            diff = (pred[i] - data.target[i]) / data.err[i]
            s_loss += 0.5 * diff * diff
        end

        tpc, _, _, _ = compute_light_tpc(data, theta)
        tpc_loss = 0.5 * ((tpc - data.tpc_target) / data.tpc_sigma)^2
        thermo_penalty = 1e4 * max(0.0, -min_sT3)^2 + 1e4 * max_phi_violation^2
        total = s_loss + tpc_loss + thermo_penalty
        if !isfinite(total)
            total = 1e30
        end
        return (loss = total, s_loss = s_loss, tpc_loss = tpc_loss, tpc = tpc, pred = pred, residuals = residuals, phis = phis)
    catch err
        return (loss = 1e30, s_loss = 1e30, tpc_loss = 1e30, tpc = NaN, pred = Float64[], residuals = Float64[], phis = zeros(0, 0))
    end
end

function save_fit_csv(path::String, data::LogUFitData, theta::AbstractVector, eval)
    open(path, "w") do io
        println(io, "T_MeV,sT3_target,sT3_pred,err,residual_over_err,phi_u,phi_d,phi_s,Phi,PhiBar,solve_residual")
        for i in eachindex(data.Ts)
            pull = (eval.pred[i] - data.target[i]) / data.err[i]
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                data.Ts[i], data.target[i], eval.pred[i], data.err[i], pull,
                eval.phis[i, 1], eval.phis[i, 2], eval.phis[i, 3], eval.phis[i, 4], eval.phis[i, 5], eval.residuals[i])
        end
    end
end

function save_tpc_csv(path::String, data::LogUFitData, theta::AbstractVector)
    tpc, peak, dlight, phis = compute_light_tpc(data, theta)
    open(path, "w") do io
        println(io, "T_MeV,phi_u,phi_d,phi_s,Phi,PhiBar,dphi_light_dT_MeV")
        for i in eachindex(data.tpc_grid)
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                data.tpc_grid[i], phis[i, 1], phis[i, 2], phis[i, 3], phis[i, 4], phis[i, 5], dlight[i])
        end
    end
    return tpc, peak
end

function save_fit_plot(path::String, data::LogUFitData, theta::AbstractVector, eval)
    title_text = @sprintf("log U fit | a1=%.5g, a2=%.5g, T0=%.3f MeV, Tpc=%.2f MeV",
        theta[1], theta[2], theta[3], eval.tpc)
    plt = scatter(
        data.Ts,
        data.target,
        yerror = data.err,
        label = "LQCD",
        xlabel = "T (MeV)",
        ylabel = "s/T^3",
        title = title_text,
        markersize = 4,
        legend = :bottomright,
    )
    plot!(plt, data.Ts, eval.pred, linewidth = 2, label = "log U model")
    savefig(plt, path)
end

function fit_logU(data::LogUFitData, theta0::Vector{Float64}; maxiters::Int)
    u0 = theta_to_u(theta0, data)
    best = Ref((loss = Inf, theta = copy(theta0), eval = evaluate_theta(theta0, data)))
    counter = Ref(0)

    function objective(u, p)
        theta = u_to_theta(u, data)
        ev = evaluate_theta(theta, data)
        counter[] += 1
        if ev.loss < best[].loss
            best[] = (loss = ev.loss, theta = copy(theta), eval = ev)
            @printf("best eval=%3d | total=% .8f s_loss=% .8f tpc=%.4f theta=(%.8f, %.8f, %.6f)\n",
                counter[], ev.loss, ev.s_loss, ev.tpc, theta[1], theta[2], theta[3])
        else
            @printf("eval=%3d      | total=% .8f s_loss=% .8f tpc=%.4f theta=(%.8f, %.8f, %.6f)\n",
                counter[], ev.loss, ev.s_loss, ev.tpc, theta[1], theta[2], theta[3])
        end
        return ev.loss
    end

    prob = Optimization.OptimizationProblem(Optimization.OptimizationFunction(objective), u0, nothing)
    result = Optimization.solve(prob, NelderMead(); maxiters = maxiters)

    theta_final = u_to_theta(result.u, data)
    ev_final = evaluate_theta(theta_final, data)
    if ev_final.loss < best[].loss
        best[] = (loss = ev_final.loss, theta = copy(theta_final), eval = ev_final)
    end
    return result, best[]
end

function run()
    csv_path = default_sT3_path()
    p_num = parse_int_arg("--p_num", 70)
    maxiters = parse_int_arg("--maxiters", 60)

    a1_start = parse_float_arg("--a1", 15.0)
    a2_start = parse_float_arg("--a2", 0.30)
    T0_start = parse_float_arg("--T0", 175.0)

    a1lo = parse_float_arg("--a1lo", 0.1)
    a1hi = parse_float_arg("--a1hi", 60.0)
    a2lo = parse_float_arg("--a2lo", 0.001)
    a2hi = parse_float_arg("--a2hi", 6.0)
    T0lo = parse_float_arg("--T0lo", 120.0)
    T0hi = parse_float_arg("--T0hi", 240.0)

    tpc_target = parse_float_arg("--tpc", 160.0)
    tpc_sigma = parse_float_arg("--tpc_sigma", 3.0)
    tpc_min = parse_float_arg("--tpc_min", 130.0)
    tpc_max = parse_float_arg("--tpc_max", 190.0)
    tpc_step = parse_float_arg("--tpc_step", 2.0)

    theta_lo = [a1lo, a2lo, T0lo]
    theta_hi = [a1hi, a2hi, T0hi]
    theta0 = clamp.([a1_start, a2_start, T0_start], theta_lo .+ 1e-8, theta_hi .- 1e-8)
    data = build_fit_data(
        csv_path = csv_path,
        p_num = p_num,
        theta_lo = theta_lo,
        theta_hi = theta_hi,
        tpc_target = tpc_target,
        tpc_sigma = tpc_sigma,
        tpc_min = tpc_min,
        tpc_max = tpc_max,
        tpc_step = tpc_step,
    )

    println("Training logarithmic Polyakov potential")
    println("U/T^4 = -a(T)/2 Phi PhiBar + b3 (T0/T)^3 log(H)")
    println("a(T) = a0 + a1 exp(-a2 T/T0) T0/T")
    @printf("fixed: a0=%.8g, b3=%.8g\n", a0, b3)
    @printf("bounds: a1=[%.6g, %.6g], a2=[%.6g, %.6g], T0=[%.3f, %.3f] MeV\n",
        theta_lo[1], theta_hi[1], theta_lo[2], theta_hi[2], theta_lo[3], theta_hi[3])
    @printf("start:  a1=%.12g, a2=%.12g, T0=%.6f MeV\n", theta0[1], theta0[2], theta0[3])
    @printf("Tpc target: %.3f MeV, sigma=%.3f MeV, grid %.2f:%.2f:%.2f\n",
        tpc_target, tpc_sigma, tpc_min, tpc_step, tpc_max)

    initial = evaluate_theta(theta0, data)
    @printf("initial total=%.12g, s_loss=%.12g, tpc=%.6f, tpc_loss=%.12g\n",
        initial.loss, initial.s_loss, initial.tpc, initial.tpc_loss)

    result, best = fit_logU(data, theta0; maxiters = maxiters)
    theta = best.theta
    ev = best.eval

    out_dir = joinpath(@__DIR__, "data")
    mkpath(out_dir)
    tag = @sprintf("logU_a1_%s_a2_%s_T0_%s", tag_param(theta[1]), tag_param(theta[2]), tag_param(theta[3]))
    out_csv = joinpath(out_dir, "$(tag).csv")
    out_tpc = joinpath(out_dir, "$(tag)_tpc.csv")
    out_plot = joinpath(out_dir, "$(tag).svg")
    save_fit_csv(out_csv, data, theta, ev)
    tpc, peak = save_tpc_csv(out_tpc, data, theta)
    save_fit_plot(out_plot, data, theta, ev)

    @printf(">> best total = %.12f\n", ev.loss)
    @printf("   s_loss = %.12f\n", ev.s_loss)
    @printf("   tpc_loss = %.12f\n", ev.tpc_loss)
    @printf("   a1 = %.12f\n", theta[1])
    @printf("   a2 = %.12f\n", theta[2])
    @printf("   T0 = %.12f MeV\n", theta[3])
    @printf("   Tpc_light = %.12f MeV, peak=%.12g\n", tpc, peak)
    println("   retcode: ", result.retcode)
    println(">> saved fit csv: ", out_csv)
    println(">> saved tpc csv: ", out_tpc)
    println(">> saved plot: ", out_plot)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
