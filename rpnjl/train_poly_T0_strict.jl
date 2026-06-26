#!/usr/bin/env julia

using Printf
using DelimitedFiles
using LinearAlgebra
using ForwardDiff
using Optimization
using OptimizationOptimJL: NelderMead

include(joinpath(@__DIR__, "rpnjl.jl"))

const POLY_A0 = 6.75
const POLY_B3 = 0.805
const POLY_B4 = 7.555

struct EosPoint
    T::Float64
    sT3::Float64
    err::Float64
    source::String
    region::String
    split::String
end

struct StrictFitData
    points::Vector{EosPoint}
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

function theta_to_u(theta::AbstractVector, data::StrictFitData)
    return [logit_clamped((theta[i] - data.theta_lo[i]) / (data.theta_hi[i] - data.theta_lo[i])) for i in eachindex(theta)]
end

function u_to_theta(u::AbstractVector, data::StrictFitData)
    theta = Vector{Float64}(undef, length(u))
    for i in eachindex(u)
        s = sigmoid_stable(u[i])
        theta[i] = data.theta_lo[i] + (data.theta_hi[i] - data.theta_lo[i]) * s
    end
    return theta
end

function region_for_temperature(T::Float64)
    if T < 130
        return "hrg_low"
    elseif T <= 155
        return "lqcd_low"
    elseif T <= 200
        return "crossover"
    elseif T <= 280
        return "mid"
    else
        return "high"
    end
end

function split_for_index(i::Int)
    r = mod(i - 1, 10)
    if r <= 5
        return "train"
    elseif r <= 7
        return "val"
    end
    return "test"
end

function load_hrg_points(path::AbstractString; Tmin::Float64, Tmax::Float64, rel_err::Float64, abs_err::Float64)
    raw = readdlm(path, skipstart = 1)
    points = EosPoint[]
    for i in axes(raw, 1)
        T = Float64(raw[i, 1])
        if Tmin <= T < Tmax
            sT3 = Float64(raw[i, 4])
            err = max(abs_err, rel_err * abs(sT3))
            push!(points, EosPoint(T, sT3, err, "HRG", region_for_temperature(T), ""))
        end
    end
    return points
end

function load_lqcd_points(path::AbstractString)
    raw = readdlm(path, ',', skipstart = 1)
    points = EosPoint[]
    for i in axes(raw, 1)
        T = Float64(raw[i, 1])
        sT3 = Float64(raw[i, 8])
        err = Float64(raw[i, 9])
        push!(points, EosPoint(T, sT3, err, "LQCD", region_for_temperature(T), ""))
    end
    return points
end

function assign_splits(points::Vector{EosPoint})
    out = EosPoint[]
    for region in ["hrg_low", "lqcd_low", "crossover", "mid", "high"]
        region_points = sort(filter(p -> p.region == region, points), by = p -> p.T)
        for (i, p) in enumerate(region_points)
            push!(out, EosPoint(p.T, p.sT3, p.err, p.source, p.region, split_for_index(i)))
        end
    end
    return sort(out, by = p -> p.T)
end

function default_hrg_path()
    return joinpath(dirname(@__DIR__), "fit_data", "HRG.txt")
end

function default_lqcd_path()
    return joinpath(dirname(@__DIR__), "fit_data", "hotqcd_1407_6387_table1_eos_origin_yerr.csv")
end

function build_fit_data(;
    hrg_path::String,
    lqcd_path::String,
    p_num::Int,
    theta_lo::Vector{Float64},
    theta_hi::Vector{Float64},
    hrg_tmin::Float64,
    hrg_tmax::Float64,
    hrg_rel_err::Float64,
    hrg_abs_err::Float64,
    tpc_target::Float64,
    tpc_sigma::Float64,
    tpc_min::Float64,
    tpc_max::Float64,
    tpc_step::Float64,
)
    points = EosPoint[]
    append!(points, load_hrg_points(hrg_path; Tmin = hrg_tmin, Tmax = hrg_tmax, rel_err = hrg_rel_err, abs_err = hrg_abs_err))
    append!(points, load_lqcd_points(lqcd_path))
    points = assign_splits(points)

    x0_candidates = [
        [-1.88, -1.88, -2.60, 0.001, 0.001],
        [-1.8, -1.8, -2.2, 0.1, 0.1],
        [-1.9, -1.9, -2.7, 0.02, 0.02],
        [-1.7, -1.7, -2.4, 0.05, 0.05],
    ]

    return StrictFitData(
        points,
        get_nodes(p_num),
        x0_candidates,
        theta_lo,
        theta_hi,
        tpc_target,
        tpc_sigma,
        range_values(tpc_min, tpc_max, tpc_step),
    )
end

function calc_U_poly_T0(T, Phi1, Phi2, theta::AbstractVector)
    a1p, a2p, T0_mev = theta
    T0p = T0_mev / hc
    b2p = POLY_A0 + a1p * T0p / T * exp(-a2p * T / T0p)
    term = -b2p / 2 * Phi1 * Phi2 -
           POLY_B3 / 6 * (Phi1^3 + Phi2^3) +
           POLY_B4 / 4 * (Phi1 * Phi2)^2
    return T^4 * term
end

function Omega_poly_T0(orders, mu_B, T, ints, theta::AbstractVector)
    p1, w1 = ints[1]
    p2, w2 = ints[2]

    phi = orders[1:3]
    Phi1 = orders[4]
    Phi2 = orders[5]
    total = chiral(phi) + calc_U_poly_T0(T, Phi1, Phi2, theta)
    masses = Mass(phi)
    mu = mu_B / 3
    for flavor in 1:3
        mass = masses[flavor]
        total += calculate_vacuum_term(p1, w1, mass)
        total += calculate_thermal_term(p2, w2, mass, T, mu, Phi1, Phi2)
    end
    return total
end

function dOmega_dorder_poly_T0(orders, mu_B, T, ints, theta::AbstractVector)
    return ForwardDiff.gradient(x -> Omega_poly_T0(x, mu_B, T, ints, theta), orders)
end

function dOmega_dT_poly_T0(orders, mu_B, T, ints, theta::AbstractVector)
    return ForwardDiff.derivative(x -> Omega_poly_T0(orders, mu_B, x, ints, theta), T)
end

function quark_mu_poly_T0(X0, mu_B, T, ints, theta::AbstractVector)
    Tout = promote_type(eltype(X0), typeof(T), typeof(mu_B), eltype(theta))
    orders = X0[1:5]
    fvec = zeros(Tout, 5)
    fvec[1:5] = dOmega_dorder_poly_T0(orders, mu_B, T, ints, theta)
    return fvec
end

function solve_order_poly_T0(T, ints, x0::AbstractVector, theta::AbstractVector)
    fWrapper(Xs) = quark_mu_poly_T0(Xs, zero(T), T, ints, theta)
    x = nonlinear_zero(fWrapper, x0; ftol = 1e-11, xtol = 1e-11, iterations = 500)
    rnorm = norm(fWrapper(x))
    if !all(isfinite, x) || !isfinite(rnorm)
        error("invalid order-parameter solution")
    end
    return Float64.(x), rnorm
end

function solve_order_best_poly_T0(T, data::StrictFitData, theta::AbstractVector)
    best_x = copy(data.x0_candidates[1])
    best_norm = Inf
    for cand in data.x0_candidates
        try
            x, rnorm = solve_order_poly_T0(T, data.ints, cand, theta)
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

function dphi_dT_implicit_poly_T0(X::AbstractVector, T::Float64, ints, theta::AbstractVector)
    z0 = vcat(Float64.(X), T)
    H = ForwardDiff.hessian(z -> Omega_poly_T0(z[1:5], 0.0, z[6], ints, theta), z0)
    Hxx = Symmetric(Matrix{Float64}(H[1:5, 1:5]))
    HxT = Vector{Float64}(H[1:5, 6])
    return -(Hxx \ HxT) ./ hc
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

function compute_light_tpc(data::StrictFitData, theta::AbstractVector)
    dlight = Vector{Float64}(undef, length(data.tpc_grid))
    X, _ = solve_order_best_poly_T0(data.tpc_grid[1] / hc, data, theta)
    for i in eachindex(data.tpc_grid)
        if i > 1
            X, _ = solve_order_poly_T0(data.tpc_grid[i] / hc, data.ints, X, theta)
        end
        dX = dphi_dT_implicit_poly_T0(X, data.tpc_grid[i] / hc, data.ints, theta)
        dlight[i] = 0.5 * (dX[1] + dX[2])
    end

    k = argmax(dlight)
    tpc = data.tpc_grid[k]
    peak = dlight[k]
    if 1 < k < length(data.tpc_grid)
        tpc, peak = parabolic_peak(
            data.tpc_grid[k - 1],
            data.tpc_grid[k],
            data.tpc_grid[k + 1],
            dlight[k - 1],
            dlight[k],
            dlight[k + 1],
        )
    end
    return tpc, peak
end

function compute_predictions(data::StrictFitData, theta::AbstractVector)
    points = sort(data.points, by = p -> p.T)
    pred = Dict{Float64, Float64}()
    residuals = Dict{Float64, Float64}()
    orders = Dict{Float64, Vector{Float64}}()
    min_sT3 = Inf
    max_phi_violation = 0.0

    X, rnorm = solve_order_best_poly_T0(points[1].T / hc, data, theta)
    for (i, p) in enumerate(points)
        if i > 1
            X, rnorm = solve_order_poly_T0(p.T / hc, data.ints, X, theta)
        end
        T = p.T / hc
        s = Float64(-dOmega_dT_poly_T0(X, 0.0, T, data.ints, theta))
        sT3 = s / T^3
        pred[p.T] = sT3
        residuals[p.T] = rnorm
        orders[p.T] = copy(X)
        min_sT3 = min(min_sT3, sT3)
        max_phi_violation = max(max_phi_violation, max(0.0, -X[4], -X[5], X[4] - 1, X[5] - 1))
    end
    return pred, residuals, orders, min_sT3, max_phi_violation
end

function split_loss(points::Vector{EosPoint}, pred::Dict{Float64, Float64}, split::AbstractString)
    selected = filter(p -> p.split == split, points)
    loss = 0.0
    for p in selected
        diff = (pred[p.T] - p.sT3) / p.err
        loss += 0.5 * diff * diff
    end
    return loss, length(selected)
end

function evaluate_theta(theta::AbstractVector, data::StrictFitData)
    if any(theta .<= data.theta_lo) || any(theta .>= data.theta_hi)
        return (total = 1e30, train = 1e30, val = 1e30, test = 1e30, tpc_loss = 1e30, tpc = NaN, pred = Dict{Float64, Float64}(), residuals = Dict{Float64, Float64}(), orders = Dict{Float64, Vector{Float64}}())
    end
    try
        pred, residuals, orders, min_sT3, max_phi_violation = compute_predictions(data, theta)
        train_loss, _ = split_loss(data.points, pred, "train")
        val_loss, _ = split_loss(data.points, pred, "val")
        test_loss, _ = split_loss(data.points, pred, "test")
        tpc, _ = compute_light_tpc(data, theta)
        tpc_loss = 0.5 * ((tpc - data.tpc_target) / data.tpc_sigma)^2
        physics_penalty = 1e4 * max(0.0, -min_sT3)^2 + 1e4 * max_phi_violation^2
        total = train_loss + tpc_loss + physics_penalty
        if !isfinite(total)
            total = 1e30
        end
        return (total = total, train = train_loss, val = val_loss, test = test_loss, tpc_loss = tpc_loss, tpc = tpc, pred = pred, residuals = residuals, orders = orders)
    catch
        return (total = 1e30, train = 1e30, val = 1e30, test = 1e30, tpc_loss = 1e30, tpc = NaN, pred = Dict{Float64, Float64}(), residuals = Dict{Float64, Float64}(), orders = Dict{Float64, Vector{Float64}}())
    end
end

function fit_strict(data::StrictFitData, theta0::Vector{Float64}; maxiters::Int)
    u0 = theta_to_u(theta0, data)
    initial = evaluate_theta(theta0, data)
    best = Ref((theta = copy(theta0), eval = initial))
    counter = Ref(0)

    function objective(u, p)
        theta = u_to_theta(u, data)
        ev = evaluate_theta(theta, data)
        counter[] += 1
        if ev.total < best[].eval.total
            best[] = (theta = copy(theta), eval = ev)
            @printf("best eval=%3d total=% .8f train=% .8f val=% .8f test=% .8f tpc=%.4f theta=(%.8f, %.8f, %.6f)\n",
                counter[], ev.total, ev.train, ev.val, ev.test, ev.tpc, theta[1], theta[2], theta[3])
        else
            @printf("eval=%3d      total=% .8f train=% .8f val=% .8f test=% .8f tpc=%.4f theta=(%.8f, %.8f, %.6f)\n",
                counter[], ev.total, ev.train, ev.val, ev.test, ev.tpc, theta[1], theta[2], theta[3])
        end
        return ev.total
    end

    optf = Optimization.OptimizationFunction(objective)
    prob = Optimization.OptimizationProblem(optf, u0, nothing)
    result = Optimization.solve(prob, NelderMead(); maxiters = maxiters)

    theta_final = u_to_theta(result.u, data)
    ev_final = evaluate_theta(theta_final, data)
    if ev_final.total < best[].eval.total
        best[] = (theta = copy(theta_final), eval = ev_final)
    end
    return result, best[]
end

function save_dataset(path::String, data::StrictFitData)
    open(path, "w") do io
        println(io, "T_MeV,sT3,err,source,region,split")
        for p in data.points
            @printf(io, "%.12g,%.12g,%.12g,%s,%s,%s\n", p.T, p.sT3, p.err, p.source, p.region, p.split)
        end
    end
end

function save_predictions(path::String, data::StrictFitData, theta::AbstractVector, ev)
    open(path, "w") do io
        println(io, "T_MeV,sT3_target,sT3_pred,err,pull,source,region,split,phi_u,phi_d,phi_s,Phi,PhiBar,solve_residual")
        for p in data.points
            X = ev.orders[p.T]
            pull = (ev.pred[p.T] - p.sT3) / p.err
            @printf(io, "%.12g,%.12g,%.12g,%.12g,%.12g,%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                p.T, p.sT3, ev.pred[p.T], p.err, pull, p.source, p.region, p.split,
                X[1], X[2], X[3], X[4], X[5], ev.residuals[p.T])
        end
    end
end

function save_summary(path::String, data::StrictFitData, theta::AbstractVector, ev, retcode)
    train_n = count(p -> p.split == "train", data.points)
    val_n = count(p -> p.split == "val", data.points)
    test_n = count(p -> p.split == "test", data.points)
    open(path, "w") do io
        println(io, "metric,value")
        @printf(io, "a1,%.12g\n", theta[1])
        @printf(io, "a2,%.12g\n", theta[2])
        @printf(io, "T0_MeV,%.12g\n", theta[3])
        @printf(io, "total_loss,%.12g\n", ev.total)
        @printf(io, "train_loss,%.12g\n", ev.train)
        @printf(io, "val_loss,%.12g\n", ev.val)
        @printf(io, "test_loss,%.12g\n", ev.test)
        @printf(io, "train_chi2_per_point,%.12g\n", 2 * ev.train / train_n)
        @printf(io, "val_chi2_per_point,%.12g\n", 2 * ev.val / val_n)
        @printf(io, "test_chi2_per_point,%.12g\n", 2 * ev.test / test_n)
        @printf(io, "Tpc_light_MeV,%.12g\n", ev.tpc)
        @printf(io, "Tpc_loss,%.12g\n", ev.tpc_loss)
        println(io, "retcode,$retcode")
        println(io, "n_train,$train_n")
        println(io, "n_val,$val_n")
        println(io, "n_test,$test_n")
    end
end

function run()
    p_num = parse_int_arg("--p_num", 70)
    maxiters = parse_int_arg("--maxiters", 80)

    a1_start = parse_float_arg("--a1", -7.796309277317753)
    a2_start = parse_float_arg("--a2", 0.11665983548204843)
    T0_start = parse_float_arg("--T0", 175.0)

    theta_lo = [
        parse_float_arg("--a1lo", -12.0),
        parse_float_arg("--a2lo", 0.02),
        parse_float_arg("--T0lo", 145.0),
    ]
    theta_hi = [
        parse_float_arg("--a1hi", -4.0),
        parse_float_arg("--a2hi", 0.35),
        parse_float_arg("--T0hi", 210.0),
    ]

    data = build_fit_data(
        hrg_path = default_hrg_path(),
        lqcd_path = default_lqcd_path(),
        p_num = p_num,
        theta_lo = theta_lo,
        theta_hi = theta_hi,
        hrg_tmin = parse_float_arg("--hrg_tmin", 50.0),
        hrg_tmax = parse_float_arg("--hrg_tmax", 130.0),
        hrg_rel_err = parse_float_arg("--hrg_rel_err", 0.50),
        hrg_abs_err = parse_float_arg("--hrg_abs_err", 0.20),
        tpc_target = parse_float_arg("--tpc", 160.0),
        tpc_sigma = parse_float_arg("--tpc_sigma", 5.0),
        tpc_min = parse_float_arg("--tpc_min", 130.0),
        tpc_max = parse_float_arg("--tpc_max", 190.0),
        tpc_step = parse_float_arg("--tpc_step", 2.0),
    )

    theta0 = clamp.([a1_start, a2_start, T0_start], theta_lo .+ 1e-8, theta_hi .- 1e-8)

    println("Strict polynomial RPNJL training with train/val/test splits")
    println("b2(T) = a0 + a1 * T0/T * exp(-a2*T/T0)")
    @printf("fixed polynomial constants: a0=%.6f, b3=%.6f, b4=%.6f\n", POLY_A0, POLY_B3, POLY_B4)
    @printf("p_num=%d, points=%d\n", p_num, length(data.points))
    @printf("start: a1=%.12f, a2=%.12f, T0=%.6f MeV\n", theta0[1], theta0[2], theta0[3])
    @printf("bounds: a1=[%.6f, %.6f], a2=[%.6f, %.6f], T0=[%.3f, %.3f] MeV\n",
        theta_lo[1], theta_hi[1], theta_lo[2], theta_hi[2], theta_lo[3], theta_hi[3])

    initial = evaluate_theta(theta0, data)
    @printf("initial total=%.12f train=%.12f val=%.12f test=%.12f Tpc=%.6f\n",
        initial.total, initial.train, initial.val, initial.test, initial.tpc)

    result, best = fit_strict(data, theta0; maxiters = maxiters)
    theta = best.theta
    ev = best.eval

    out_dir = joinpath(@__DIR__, "data", "strict_poly_T0")
    mkpath(out_dir)
    tag = @sprintf("polyT0_a1_%s_a2_%s_T0_%s", tag_param(theta[1]), tag_param(theta[2]), tag_param(theta[3]))
    dataset_path = joinpath(out_dir, "dataset_split.csv")
    pred_path = joinpath(out_dir, "$(tag)_predictions.csv")
    summary_path = joinpath(out_dir, "$(tag)_summary.csv")

    save_dataset(dataset_path, data)
    save_predictions(pred_path, data, theta, ev)
    save_summary(summary_path, data, theta, ev, result.retcode)

    @printf(">> best total = %.12f\n", ev.total)
    @printf("   train loss = %.12f\n", ev.train)
    @printf("   val loss   = %.12f\n", ev.val)
    @printf("   test loss  = %.12f\n", ev.test)
    @printf("   a1 = %.12f\n", theta[1])
    @printf("   a2 = %.12f\n", theta[2])
    @printf("   T0 = %.12f MeV\n", theta[3])
    @printf("   Tpc_light = %.12f MeV\n", ev.tpc)
    println("   retcode: ", result.retcode)
    println(">> saved dataset split: ", dataset_path)
    println(">> saved predictions: ", pred_path)
    println(">> saved summary: ", summary_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
