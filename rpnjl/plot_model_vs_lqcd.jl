#!/usr/bin/env julia

using Plots

include(joinpath(@__DIR__, "rpnjl.jl"))
include(joinpath(@__DIR__, "train_a1_a2.jl"))

"""
    run_compare(; a1, a2, csv_path, p_num)

    用给定 a1/a2 计算 s/T^3，并与 sT3.csv 进行对比。
    返回 (Ts, target, pred, err, loss)。
"""
function run_compare(;
    a1::Float64 = -9.809526970026672,
    a2::Float64 = 0.2615358295915334,
    csv_path::String = joinpath(@__DIR__, "sT3.csv"),
    p_num::Int = 200,
)
    Ts, target, err = load_sT3_csv(csv_path)
    x0_init = solve_Tmu_precise_130(
        csv_path = csv_path,
        a1_0 = a1,
        a2_0 = a2,
        p_num = p_num,
        jitter = 0.12,
        n_restarts = 20
    )

    ints = get_nodes(p_num)
    loss, pred = sT3_fit_objective([a1, a2], Ts, target, err, ints, x0_init)

    out_path = joinpath(@__DIR__, "compare_a1_$(replace(string(a1), "." => "_")).svg")
    plt = scatter(
        Ts,
        target,
        yerror = err,
        label = "LQCD (sT3)",
        xlabel = "T [MeV]",
        ylabel = "s/T^3",
        title = "PNJL model vs LQCD\n(a1=$(a1), a2=$(a2), loss=$(round(loss, digits=4)))",
        legend = :topleft,
        markerstrokewidth = 0.5,
        markersize = 4,
        dpi = 200
    )
    plot!(plt, Ts, pred, lw = 2, label = "Model (a1/a2)")
    savefig(plt, out_path)

    println(">> 输出对比图: ", out_path)
    println(">> loss = ", loss)
    return Ts, target, pred, err, loss, out_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    # 允许命令行覆盖参数：julia plot_model_vs_lqcd.jl a1 a2
    a1v = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : -9.809526970026672
    a2v = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.2615358295915334
    run_compare(a1 = a1v, a2 = a2v)
end
