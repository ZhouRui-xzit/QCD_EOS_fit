#!/usr/bin/env julia

using Printf
using LinearAlgebra
using ForwardDiff
using Optim
using Distributions
using Turing
using Plots

include(joinpath(@__DIR__, "train_a1_a2.jl"))

struct ImplicitFitData
    Ts::Vector{Float64}
    target::Vector{Float64}
    err::Vector{Float64}
    ints::Any
    x0_candidates::Vector{Vector{Float64}}
    a1lo::Float64
    a1hi::Float64
    a2lo::Float64
    a2hi::Float64
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

sigmoid_stable(x) = x >= 0 ? inv(1 + exp(-x)) : (exp(x) / (1 + exp(x)))

function logit_clamped(x::Float64)
    xc = clamp(x, 1e-8, 1 - 1e-8)
    return log(xc / (1 - xc))
end

function theta_to_u(theta::AbstractVector, data::ImplicitFitData)
    z1 = (theta[1] - data.a1lo) / (data.a1hi - data.a1lo)
    z2 = (theta[2] - data.a2lo) / (data.a2hi - data.a2lo)
    return [logit_clamped(z1), logit_clamped(z2)]
end

function u_to_theta_and_jac(u::AbstractVector, data::ImplicitFitData)
    s1 = sigmoid_stable(u[1])
    s2 = sigmoid_stable(u[2])
    theta = [
        data.a1lo + (data.a1hi - data.a1lo) * s1,
        data.a2lo + (data.a2hi - data.a2lo) * s2,
    ]
    jac = [
        (data.a1hi - data.a1lo) * s1 * (1 - s1),
        (data.a2hi - data.a2lo) * s2 * (1 - s2),
    ]
    return theta, jac
end

function solve_order_param(T, ints, x0::AbstractVector, theta::AbstractVector)
    a1p, a2p = theta
    mu_B = zero(T)
    fWrapper(Xs) = Quark_mu_param(Xs, mu_B, T, ints, a1p, a2p)
    res = nlsolve(fWrapper, copy(x0), autodiff = :forward, ftol = 1e-12, xtol = 1e-12, iterations = 600)
    x = res.zero
    return x, norm(fWrapper(x))
end

function solve_order_best(T, data::ImplicitFitData, theta::AbstractVector)
    best_x = copy(data.x0_candidates[1])
    best_norm = Inf
    for cand in data.x0_candidates
        try
            x, rnorm = solve_order_param(T, data.ints, cand, theta)
            if all(isfinite, x) && isfinite(rnorm) && rnorm < best_norm
                best_x = Float64.(x)
                best_norm = rnorm
            end
        catch
        end
    end
    return best_x, best_norm
end

function sT3_and_grad_theta_implicit(X::AbstractVector, T::Float64, ints, theta::AbstractVector)
    mus = [0.0, 0.0, 0.0]
    z0 = vcat(Float64.(X), T, Float64.(theta))
    omega_z(z) = Omega_param(z[1:5], mus, z[6], ints, z[7], z[8])
    H = ForwardDiff.hessian(omega_z, z0)

    Hxx = Matrix{Float64}(H[1:5, 1:5])
    HxT = Vector{Float64}(H[1:5, 6])
    Hxth = Matrix{Float64}(H[1:5, 7:8])
    HTth = Vector{Float64}(H[6, 7:8])

    solve_term = Hxx \ Hxth
    ds_dtheta = -HTth + vec(transpose(HxT) * solve_term)

    s = Float64(-dOmega_dT_param(X, mus, T, ints, theta[1], theta[2]))
    return s / T^3, ds_dtheta ./ T^3
end

function loss_and_grad_theta(theta::AbstractVector, data::ImplicitFitData; need_grad::Bool = true)
    if theta[1] <= data.a1lo || theta[1] >= data.a1hi || theta[2] <= data.a2lo || theta[2] >= data.a2hi
        return Inf, zeros(2), Float64[]
    end

    loss = 0.0
    grad = zeros(2)
    pred = Vector{Float64}(undef, length(data.Ts))

    try
        X, _ = solve_order_best(data.Ts[1] / hc, data, theta)
        for i in eachindex(data.Ts)
            if i > 1
                X, _ = solve_order_param(data.Ts[i] / hc, data.ints, X, theta)
            end

            if need_grad
                pred_i, dpred = sT3_and_grad_theta_implicit(X, data.Ts[i] / hc, data.ints, theta)
                diff = (pred_i - data.target[i]) / data.err[i]
                loss += 0.5 * diff * diff
                grad .+= (diff / data.err[i]) .* dpred
                pred[i] = pred_i
            else
                s = Float64(-dOmega_dT_param(X, [0.0, 0.0, 0.0], data.Ts[i] / hc, data.ints, theta[1], theta[2]))
                pred_i = s / (data.Ts[i] / hc)^3
                diff = (pred_i - data.target[i]) / data.err[i]
                loss += 0.5 * diff * diff
                pred[i] = pred_i
            end
        end
    catch err
        return 1e30, zeros(2), Float64[]
    end

    if !isfinite(loss) || any(!isfinite, grad)
        return 1e30, zeros(2), Float64[]
    end
    return loss, grad, pred
end

@model function pnjl_turing_external_likelihood(data::ImplicitFitData; a1_mu = -7.8, a1_sigma = 0.45, a2_mu = 0.12, a2_sigma = 0.04)
    a1 ~ truncated(Normal(a1_mu, a1_sigma), data.a1lo, data.a1hi)
    a2 ~ truncated(Normal(a2_mu, a2_sigma), data.a2lo, data.a2hi)
    loss, _, _ = loss_and_grad_theta([Float64(a1), Float64(a2)], data; need_grad = false)
    Turing.@addlogprob! -loss
    return nothing
end

function build_fit_data(; csv_path::String, p_num::Int, a1lo::Float64, a1hi::Float64, a2lo::Float64, a2hi::Float64)
    Ts, target, err = load_sT3_csv(csv_path)
    ints = get_nodes(p_num)
    x0_candidates = [
        [-1.88, -1.88, -2.60, 0.001, 0.001],
        [-1.8, -1.8, -2.2, 0.1, 0.1],
        [-1.9, -1.9, -2.7, 0.02, 0.02],
        [-1.7, -1.7, -2.4, 0.05, 0.05],
    ]
    return ImplicitFitData(Ts, target, err, ints, x0_candidates, a1lo, a1hi, a2lo, a2hi)
end

function default_sT3_path()
    direct = joinpath(@__DIR__, "sT3.csv")
    nested = joinpath(@__DIR__, "data", "sT3.csv")
    return isfile(direct) ? direct : nested
end

function save_fit_csv(path::String, data::ImplicitFitData, theta::AbstractVector, pred::AbstractVector)
    open(path, "w") do io
        println(io, "T,sT3_target,sT3_pred,err")
        for i in eachindex(data.Ts)
            @printf(io, "%.12g,%.12g,%.12g,%.12g\n", data.Ts[i], data.target[i], pred[i], data.err[i])
        end
    end
end

function save_fit_plot(path::String, data::ImplicitFitData, theta::AbstractVector, pred::AbstractVector, loss::Float64)
    plt = scatter(
        data.Ts,
        data.target,
        yerror = data.err,
        label = "LQCD",
        xlabel = "T (MeV)",
        ylabel = "s/T^3",
        title = @sprintf("Implicit AD fit | a1=%.6f, a2=%.6f, loss=%.6f", theta[1], theta[2], loss),
        legend = :topright,
        markersize = 4,
    )
    plot!(plt, data.Ts, pred, linewidth = 2, label = "Model")
    savefig(plt, path)
end

function theta_from_mode_result(mode_result)
    names, values = Turing.Optimisation.vector_names_and_params(mode_result)
    theta = fill(NaN, 2)
    for (vn, value) in zip(names, values)
        name = string(vn)
        if occursin("a1", name)
            theta[1] = Float64(value)
        elseif occursin("a2", name)
            theta[2] = Float64(value)
        end
    end
    if any(!isfinite, theta)
        error("Could not extract a1/a2 from Turing ModeResult: names=$(names), values=$(values)")
    end
    return theta
end

function train_turing_map(data::ImplicitFitData, theta0::Vector{Float64}; maxiters::Int)
    model = pnjl_turing_external_likelihood(data)
    mode_result = Turing.maximum_a_posteriori(
        model;
        initial_params = (a1 = theta0[1], a2 = theta0[2]),
        lb = (a1 = data.a1lo, a2 = data.a2lo),
        ub = (a1 = data.a1hi, a2 = data.a2hi),
        adtype = Turing.Optimisation.ADTypes.AutoFiniteDiff(),
        maxiters = maxiters,
    )
    theta = theta_from_mode_result(mode_result)
    loss, _, pred = loss_and_grad_theta(theta, data; need_grad = false)
    return mode_result, (loss = loss, theta = theta, pred = pred)
end

function train_implicit(data::ImplicitFitData, theta0::Vector{Float64}; maxiter::Int)
    u0 = theta_to_u(theta0, data)
    eval_counter = Ref(0)
    best = Ref((loss = Inf, theta = copy(theta0), pred = Float64[]))

    function fg!(F, G, u)
        theta, jac = u_to_theta_and_jac(u, data)
        loss, grad_theta, pred = loss_and_grad_theta(theta, data; need_grad = true)
        eval_counter[] += 1

        if loss < best[].loss && length(pred) == length(data.Ts)
            best[] = (loss = loss, theta = copy(theta), pred = copy(pred))
            @printf("best update eval=%3d | a1=% .10f, a2=% .10f, loss=% .10f\n",
                eval_counter[], theta[1], theta[2], loss)
        else
            @printf("eval=%3d | a1=% .10f, a2=% .10f, loss=% .10f\n",
                eval_counter[], theta[1], theta[2], loss)
        end

        if G !== nothing
            G[1] = grad_theta[1] * jac[1]
            G[2] = grad_theta[2] * jac[2]
        end
        return F === nothing ? nothing : loss
    end

    result = optimize(Optim.only_fg!(fg!), u0, LBFGS(), Optim.Options(iterations = maxiter, show_trace = true, show_every = 1))
    theta_opt, _ = u_to_theta_and_jac(Optim.minimizer(result), data)
    final_loss, _, final_pred = loss_and_grad_theta(theta_opt, data; need_grad = true)

    if final_loss < best[].loss && length(final_pred) == length(data.Ts)
        best[] = (loss = final_loss, theta = copy(theta_opt), pred = copy(final_pred))
    end

    return result, best[]
end

function run()
    csv_path = default_sT3_path()
    p_num = parse_int_arg("--p_num", 120)
    maxiter = parse_int_arg("--maxiter", 20)
    mh_samples = parse_int_arg("--mh", 0)
    turing_map = parse_int_arg("--turing_map", 0)

    a1lo = parse_float_arg("--a1lo", -8.2)
    a1hi = parse_float_arg("--a1hi", -7.4)
    a2lo = parse_float_arg("--a2lo", 0.09)
    a2hi = parse_float_arg("--a2hi", 0.14)
    a1_start = parse_float_arg("--a1", -7.796309277317753)
    a2_start = parse_float_arg("--a2", 0.11665983548204843)

    data = build_fit_data(csv_path = csv_path, p_num = p_num, a1lo = a1lo, a1hi = a1hi, a2lo = a2lo, a2hi = a2hi)
    theta0 = [a1_start, a2_start]

    mode_label = turing_map == 0 ? "Optim LBFGS with implicit AD gradient" : "Turing MAP with AutoFiniteDiff"
    println("Turing v", pkgversion(Turing), " + ", mode_label)
    @printf("bounds: a1=[%.6f, %.6f], a2=[%.6f, %.6f], p_num=%d\n", a1lo, a1hi, a2lo, a2hi, p_num)
    @printf("start:  a1=%.12f, a2=%.12f\n", theta0[1], theta0[2])

    loss0, grad0, _ = loss_and_grad_theta(theta0, data; need_grad = true)
    @printf("initial loss=%.12f, grad=(%.12g, %.12g), grad_norm=%.12g\n", loss0, grad0[1], grad0[2], norm(grad0))

    result = nothing
    best = nothing
    if turing_map != 0
        println(">> running Turing.maximum_a_posteriori with AutoFiniteDiff")
        result, best = train_turing_map(data, theta0; maxiters = maxiter)
    else
        result, best = train_implicit(data, theta0; maxiter = maxiter)
    end

    @printf(">> training done\n")
    @printf("   a1 = %.12f\n", best.theta[1])
    @printf("   a2 = %.12f\n", best.theta[2])
    @printf("   loss = %.12f\n", best.loss)
    if turing_map == 0
        println("   Optim converged: ", Optim.converged(result))
    else
        println("   Turing MAP lp: ", result.lp)
    end

    out_csv = joinpath(@__DIR__, @sprintf("turing_implicit_best_a1_%s_a2_%s.csv", tag_param(best.theta[1]), tag_param(best.theta[2])))
    out_plot = joinpath(@__DIR__, @sprintf("turing_implicit_best_a1_%s_a2_%s.svg", tag_param(best.theta[1]), tag_param(best.theta[2])))
    save_fit_csv(out_csv, data, best.theta, best.pred)
    save_fit_plot(out_plot, data, best.theta, best.pred, best.loss)
    println(">> saved fit csv: ", out_csv)
    println(">> saved fit plot: ", out_plot)

    if mh_samples > 0
        println(">> running optional Turing MH samples: ", mh_samples)
        model = pnjl_turing_external_likelihood(data)
        chain = sample(model, MH(), mh_samples)
        println(chain)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
