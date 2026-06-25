#!/usr/bin/env julia

using Printf
using Random
using LinearAlgebra
using Statistics
using Plots

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

function solve_Tmu_130_for_param(T130, ints, x0_init::AbstractVector, a1p::Float64, a2p::Float64)
    try
        mu_B = zero(T130)
        fWrapper(Xs) = Quark_mu_param(Xs, mu_B, T130, ints, a1p, a2p)
        x_ok = nonlinear_zero(fWrapper, x0_init; ftol = 1e-12, xtol = 1e-12, iterations = 400)
        if all(isfinite, x_ok)
            return x_ok
        end
    catch
    end
    return x0_init
end

function objective_for_param(a1::Float64, a2::Float64, Ts, target, err, ints, x0_init::AbstractVector)
    T130 = Ts[1] / hc
    x0 = solve_Tmu_130_for_param(T130, ints, x0_init, a1, a2)
    try
        loss, pred = sT3_fit_objective([a1, a2], Ts, target, err, ints, x0)
        return Float64(loss), pred
    catch
        return Inf, Float64[]
    end
end

function tag_param(x::Real)
    s = @sprintf("%.6f", x)
    return replace(s, "-" => "m", "+" => "p", "." => "p")
end

function to_unit(a1::Float64, a2::Float64, bounds)
    return [
        (a1 - bounds.a1min) / (bounds.a1max - bounds.a1min),
        (a2 - bounds.a2min) / (bounds.a2max - bounds.a2min),
    ]
end

function from_unit(u::AbstractVector, bounds)
    a1 = bounds.a1min + clamp(u[1], 0.0, 1.0) * (bounds.a1max - bounds.a1min)
    a2 = bounds.a2min + clamp(u[2], 0.0, 1.0) * (bounds.a2max - bounds.a2min)
    return a1, a2
end

function points_matrix(points::Vector{Vector{Float64}})
    X = Matrix{Float64}(undef, length(points), 2)
    for i in eachindex(points)
        X[i, 1] = points[i][1]
        X[i, 2] = points[i][2]
    end
    return X
end

function lhs_points(n::Int, rng::AbstractRNG)
    X = Matrix{Float64}(undef, n, 2)
    for d in 1:2
        vals = ((0:n-1) .+ rand(rng, n)) ./ n
        X[:, d] .= shuffle(rng, vals)
    end
    return X
end

function rbf_kernel(X1::AbstractMatrix, X2::AbstractMatrix, ell::Float64)
    K = Matrix{Float64}(undef, size(X1, 1), size(X2, 1))
    @inbounds for i in axes(X1, 1), j in axes(X2, 1)
        dx1 = (X1[i, 1] - X2[j, 1]) / ell
        dx2 = (X1[i, 2] - X2[j, 2]) / ell
        K[i, j] = exp(-0.5 * (dx1 * dx1 + dx2 * dx2))
    end
    return K
end

function fit_gp(X::AbstractMatrix, y::AbstractVector; ell::Float64 = 0.28, noise::Float64 = 1e-6)
    ymean = mean(y)
    yscale = std(y)
    if !isfinite(yscale) || yscale <= 1e-12
        yscale = 1.0
    end
    yz = (y .- ymean) ./ yscale

    K0 = rbf_kernel(X, X, ell)
    for jitter in (noise, 1e-8, 1e-6, 1e-4, 1e-2)
        try
            K = K0 + jitter * I
            chol = cholesky(Symmetric(K))
            alpha = chol \ yz
            return (X = Matrix{Float64}(X), y = collect(y), yz = yz, ymean = ymean, yscale = yscale,
                    ell = ell, chol = chol, alpha = alpha)
        catch
        end
    end
    error("GP Cholesky failed")
end

function gp_predict_z(model, Xstar::AbstractMatrix)
    Ks = rbf_kernel(model.X, Xstar, model.ell)
    mu = vec(transpose(Ks) * model.alpha)
    v = model.chol \ Ks
    var = ones(size(Xstar, 1)) .- vec(sum(Ks .* v; dims = 1))
    sigma = sqrt.(max.(var, 1e-12))
    return mu, sigma
end

normal_pdf(z) = inv(sqrt(2pi)) * exp(-0.5 * z * z)
function normal_cdf(z::Real)
    x = Float64(z)
    ax = abs(x)
    t = inv(1.0 + 0.2316419 * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    cdf_pos = 1.0 - normal_pdf(ax) * poly
    return x >= 0 ? cdf_pos : 1.0 - cdf_pos
end

function expected_improvement(mu::Float64, sigma::Float64, best_z::Float64, xi::Float64)
    if sigma <= 1e-12
        return 0.0
    end
    improve = best_z - mu - xi
    z = improve / sigma
    return improve * normal_cdf(z) + sigma * normal_pdf(z)
end

function min_dist2_to_existing(u1::Float64, u2::Float64, points::Vector{Vector{Float64}})
    best = Inf
    for p in points
        d1 = u1 - p[1]
        d2 = u2 - p[2]
        best = min(best, d1 * d1 + d2 * d2)
    end
    return best
end

function propose_next(unit_points, losses, rng; candidates::Int, ell::Float64, noise::Float64, xi::Float64)
    X = points_matrix(unit_points)
    model = fit_gp(X, losses; ell = ell, noise = noise)
    best_idx = argmin(losses)
    best_u = unit_points[best_idx]

    n_global = max(1, round(Int, candidates * 0.65))
    n_local = candidates - n_global
    cand = Matrix{Float64}(undef, candidates, 2)
    cand[1:n_global, :] .= rand(rng, n_global, 2)
    for i in 1:n_local
        row = n_global + i
        cand[row, 1] = clamp(best_u[1] + 0.16 * randn(rng), 0.0, 1.0)
        cand[row, 2] = clamp(best_u[2] + 0.16 * randn(rng), 0.0, 1.0)
    end

    mu, sigma = gp_predict_z(model, cand)
    best_z = minimum(model.yz)
    best_ei = -Inf
    best_row = 1
    for i in axes(cand, 1)
        if min_dist2_to_existing(cand[i, 1], cand[i, 2], unit_points) < 1e-8
            continue
        end
        ei = expected_improvement(mu[i], sigma[i], best_z, xi)
        if ei > best_ei
            best_ei = ei
            best_row = i
        end
    end
    return [cand[best_row, 1], cand[best_row, 2]], best_ei
end

function plot_best(Ts, target, err, pred, a1, a2, loss; out_dir::AbstractString = @__DIR__)
    if length(pred) != length(Ts) || isempty(pred)
        return ""
    end
    out_path = joinpath(out_dir, @sprintf("bayes_best_a1_%s_a2_%s.svg", tag_param(a1), tag_param(a2)))
    plt = scatter(
        Ts,
        target,
        yerror = err,
        label = "LQCD",
        xlabel = "T (MeV)",
        ylabel = "s/T^3",
        title = @sprintf("Bayes opt best | a1=%.6f, a2=%.6f, loss=%.6f", a1, a2, loss),
        legend = :topright,
        markersize = 4,
    )
    plot!(plt, Ts, pred, linewidth = 2, label = "Model")
    savefig(plt, out_path)
    return out_path
end

function run_bayes_opt()
    csv_path = joinpath(@__DIR__, "sT3.csv")
    p_num = parse_int_arg("--p_num", 120)
    seed = parse_int_arg("--seed", 20260623)
    n_init = parse_int_arg("--init", 8)
    n_iter = parse_int_arg("--iter", 24)
    candidates = parse_int_arg("--candidates", 1600)

    bounds = (
        a1min = parse_float_arg("--a1min", -9.8),
        a1max = parse_float_arg("--a1max", -8.6),
        a2min = parse_float_arg("--a2min", 0.15),
        a2max = parse_float_arg("--a2max", 0.28),
    )
    center_a1 = parse_float_arg("--center_a1", -9.2)
    center_a2 = parse_float_arg("--center_a2", 0.215)
    ell = parse_float_arg("--ell", 0.28)
    noise = parse_float_arg("--noise", 1e-6)
    xi = parse_float_arg("--xi", 0.01)

    rng = MersenneTwister(seed)
    Ts, target, err = load_sT3_csv(csv_path)
    ints = get_nodes(p_num)
    x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1]

    println("Bayesian optimization for a1/a2")
    @printf("bounds: a1=[%.6f, %.6f], a2=[%.6f, %.6f]\n", bounds.a1min, bounds.a1max, bounds.a2min, bounds.a2max)
    @printf("settings: p_num=%d, init=%d, iter=%d, candidates=%d, seed=%d\n", p_num, n_init, n_iter, candidates, seed)

    unit_points = Vector{Vector{Float64}}()
    losses = Float64[]
    rows = Vector{Tuple{Int,String,Float64,Float64,Float64}}()
    best = Ref((a1 = NaN, a2 = NaN, loss = Inf, pred = Float64[]))
    eval_id = Ref(0)

    function evaluate!(u::Vector{Float64}, stage::String)
        eval_id[] += 1
        a1, a2 = from_unit(u, bounds)
        loss, pred = objective_for_param(a1, a2, Ts, target, err, ints, x0_init)
        push!(unit_points, copy(u))
        push!(losses, loss)
        push!(rows, (eval_id[], stage, a1, a2, loss))
        @printf("%-5s eval=%3d | a1=% .8f, a2=% .8f -> loss=% .8f\n", stage, eval_id[], a1, a2, loss)
        if loss < best[].loss
            best[] = (a1 = a1, a2 = a2, loss = loss, pred = pred)
            @printf("      best update: a1=% .8f, a2=% .8f, loss=% .8f\n", a1, a2, loss)
        end
    end

    evaluate!(to_unit(center_a1, center_a2, bounds), "init")
    init_rest = max(0, n_init - 1)
    init_X = lhs_points(init_rest, rng)
    for i in axes(init_X, 1)
        evaluate!([init_X[i, 1], init_X[i, 2]], "init")
    end

    for it in 1:n_iter
        u_next, ei = propose_next(unit_points, losses, rng; candidates = candidates, ell = ell, noise = noise, xi = xi)
        a1_next, a2_next = from_unit(u_next, bounds)
        @printf("bo    step=%3d | next a1=% .8f, a2=% .8f, EI=%.8g\n", it, a1_next, a2_next, ei)
        evaluate!(u_next, "bo")
    end

    sort_rows = sort(rows, by = r -> r[5])
    println(">> top candidates")
    for k in 1:min(10, length(sort_rows))
        r = sort_rows[k]
        @printf("  %2d) eval=%3d %-5s a1=% .8f, a2=% .8f, loss=% .8f\n", k, r[1], r[2], r[3], r[4], r[5])
    end

    out_csv = joinpath(@__DIR__, "bayes_opt_a1_a2.csv")
    open(out_csv, "w") do io
        println(io, "eval,stage,a1,a2,loss")
        for r in rows
            @printf(io, "%d,%s,%.12g,%.12g,%.12g\n", r[1], r[2], r[3], r[4], r[5])
        end
    end

    out_fit = joinpath(@__DIR__, "bayes_best_fit.csv")
    open(out_fit, "w") do io
        println(io, "T,sT3_target,pred_best,error")
        for i in eachindex(Ts)
            if length(best[].pred) == length(Ts)
                @printf(io, "%.12g,%.12g,%.12g,%.12g\n", Ts[i], target[i], best[].pred[i], err[i])
            else
                @printf(io, "%.12g,%.12g,,%.12g\n", Ts[i], target[i], err[i])
            end
        end
    end

    out_plot = plot_best(Ts, target, err, best[].pred, best[].a1, best[].a2, best[].loss; out_dir = @__DIR__)
    println(">> best overall: a1=$(best[].a1), a2=$(best[].a2), loss=$(best[].loss)")
    println(">> saved BO evaluations: ", out_csv)
    println(">> saved best fit csv: ", out_fit)
    println(">> saved best plot: ", out_plot)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_bayes_opt()
end
