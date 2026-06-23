#!/usr/bin/env julia

using Printf
using Plots

include(joinpath(@__DIR__, "train_a1_a2.jl"))

"""
    parse_float_arg(key, default)
"""
function parse_float_arg(key::AbstractString, default::Float64)
    for i in 1:length(ARGS)-1
        if ARGS[i] == key
            return parse(Float64, ARGS[i + 1])
        end
    end
    return default
end

"""
    parse_int_arg(key, default)
"""
function parse_int_arg(key::AbstractString, default::Int)
    for i in 1:length(ARGS)-1
        if ARGS[i] == key
            return parse(Int, ARGS[i + 1])
        end
    end
    return default
end

"""
    solve_Tmu_130_for_param(T130, a1p, a2p, ints, x0_init)
"""
function solve_Tmu_130_for_param(T130, ints, x0_init::AbstractVector, a1p::Float64, a2p::Float64)
    try
        mu_B = zero(T130)
        fWrapper(Xs) = Quark_mu_param(Xs, mu_B, T130, ints, a1p, a2p)
        res = nlsolve(fWrapper, copy(x0_init), autodiff = :forward, ftol = 1e-12, xtol = 1e-12, iterations = 400)
        x_ok = res.zero
        if all(isfinite, x_ok)
            return x_ok
        end
    catch
    end
    return x0_init
end

"""
    grid_objective(a1,a2,Ts,target,err,ints,x0_init)
"""
function grid_objective(a1::Float64, a2::Float64, Ts, target, err, ints, x0_init::AbstractVector)
    T130 = Ts[1] / hc
    x0 = solve_Tmu_130_for_param(T130, ints, x0_init, a1, a2)
    try
        loss, pred = sT3_fit_objective([a1, a2], Ts, target, err, ints, x0)
        return Float64(loss), pred
    catch
        return Inf, Float64[]
    end
end

"""
    _tag_param(x)
"""
function _tag_param(x::Real)
    s = @sprintf("%.6f", x)
    return replace(s, "-" => "m", "+" => "p", "." => "p")
end

"""
    plot_best_result(Ts, target, err, pred, a1, a2, loss; out_dir=@__DIR__)
"""
function plot_best_result(Ts, target, err, pred, a1, a2, loss; out_dir::AbstractString = @__DIR__)
    if length(pred) != length(Ts)
        return
    end
    if isempty(pred) || all(!isfinite, pred)
        return
    end

    fname = @sprintf("grid_best_a1_%s_a2_%s.svg", _tag_param(a1), _tag_param(a2))
    fpath = joinpath(out_dir, fname)

    p = plot(
        Ts,
        target,
        seriestype = :scatter,
        yerror = err,
        label = "LQCD",
        markersize = 4,
        xlabel = "T (MeV)",
        ylabel = "s/T^3",
        title = @sprintf("final best | a1=%.6f, a2=%.6f, loss=%.6f", a1, a2, loss),
        legend = :topright,
    )
    plot!(p, Ts, pred, linewidth = 2, label = "Model")
    savefig(p, fpath)
    @printf(">> saved best plot: %s\n", fpath)
end

"""
    run_grid_search()
"""
function run_grid_search()
    csv_path = joinpath(@__DIR__, "sT3.csv")
    p_num = parse_int_arg("--p_num", 120)
    a1min = parse_float_arg("--a1min", -10.6)
    a1max = parse_float_arg("--a1max", -9.0)
    a1n  = parse_int_arg("--a1n", 9)
    a2min = parse_float_arg("--a2min", 0.02)
    a2max = parse_float_arg("--a2max", 0.8)
    a2n  = parse_int_arg("--a2n", 17)

    Ts, target, err = load_sT3_csv(csv_path)
    ints = get_nodes(p_num)
    x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1]

    a1_values = collect(range(a1min, a1max; length = a1n))
    a2_values = collect(range(a2min, a2max; length = a2n))

    nA = length(a1_values)
    nB = length(a2_values)
    npoints = nA * nB
    println("Grid search: a1=$nA, a2=$nB, total=$npoints")

    header = ["a1", "a2", "loss"]
    rows = Vector{Vector{Float64}}()

    best = (a1 = a1_values[1], a2 = a2_values[1], loss = Inf, pred = Float64[])
    counter = 0

    for a1 in a1_values
        for a2 in a2_values
            counter += 1
            @printf("scan %5d/%d | a1=% .6f, a2=% .6f\n", counter, npoints, a1, a2)
            loss, pred = grid_objective(a1, a2, Ts, target, err, ints, x0_init)
            @printf("          -> loss=%.6f\n", loss)

            push!(rows, [a1, a2, loss])
            if loss < best.loss
                best = (a1 = a1, a2 = a2, loss = loss, pred = pred)
                @printf("             best update: a1=% .6f, a2=% .6f, loss=% .6f\n", a1, a2, loss)
            end
        end
    end

    sort_rows = sort(rows, by = r -> r[3])
    println(">> best overall: (a1=$(best.a1), a2=$(best.a2), loss=$(best.loss))")
    for k in 1:min(10, length(sort_rows))
        @printf("  %2d) a1=% .6f, a2=% .6f, loss=% .6f\n", k, sort_rows[k][1], sort_rows[k][2], sort_rows[k][3])
    end

    out_csv = joinpath(@__DIR__, "grid_search_a1_a2.csv")
    open(out_csv, "w") do io
        println(io, join(header, ","))
        for r in rows
            @printf(io, "%.12g,%.12g,%.12g\n", r[1], r[2], r[3])
        end
    end

    out_top1 = joinpath(@__DIR__, "grid_best_fit.csv")
    open(out_top1, "w") do io
        println(io, "T,sT3_target,pred_best,error")
        for i in eachindex(Ts)
            if length(best.pred) == length(Ts)
                @printf(io, "%.12g,%.12g,%.12g,%.12g\n", Ts[i], target[i], best.pred[i], err[i])
            else
                @printf(io, "%.12g,%.12g,,%.12g\n", Ts[i], target[i], err[i])
            end
        end
    end

    println(">> saved grid results: ", out_csv)
    println(">> saved best fit csv: ", out_top1)
    println(">> total trained points: ", counter)
    plot_best_result(Ts, target, err, best.pred, best.a1, best.a2, best.loss; out_dir = @__DIR__)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_grid_search()
end
