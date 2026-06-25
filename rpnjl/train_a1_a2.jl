#!/usr/bin/env julia

using DelimitedFiles
using Random
using Printf

include(joinpath(@__DIR__, "rpnjl.jl"))

"""
    run_train(; kwargs...)

    独立训练入口：调用 rpnjl.jl 中的 train_a1_a2，返回最佳 a1/a2 与拟合结果。
"""
function run_train(;
    csv_path::String = joinpath(@__DIR__, "sT3.csv"),
    x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1],
    a1_0::Float64 = -9.8,
    a2_0::Float64 = 0.26,
    maxiter::Int = 80,
    lr::Float64 = 0.05,
    p_num::Int = 200,
    grad_eps::Float64 = 1e-8,
)
    result = train_a1_a2(;
        csv_path = csv_path,
        x0_init = x0_init,
        a1_0 = a1_0,
        a2_0 = a2_0,
        maxiter = maxiter,
        lr = lr,
        p_num = p_num,
        verbose = true,
        grad_eps = grad_eps,
    )
    return result
end

"""
    perturb_and_train(;
        csv_path = "sT3.csv",
        x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1],
        a1_0::Float64 = -9.8,
        a2_0::Float64 = 0.26,
        maxiter::Int = 80,
        lr::Float64 = 0.01,
        p_num::Int = 200,
        grad_eps::Float64 = 1e-7,
        n_restarts::Int = 5,
        jitter_frac::Float64 = 0.05,
    )

    在默认值附近生成多次重启，选取最优结果。
"""
function perturb_and_train(;
    csv_path::String = joinpath(@__DIR__, "sT3.csv"),
    x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1],
    a1_0::Float64 = -9.8,
    a2_0::Float64 = 0.26,
    maxiter::Int = 80,
    lr::Float64 = 0.01,
    p_num::Int = 200,
    grad_eps::Float64 = 1e-7,
    n_restarts::Int = 5,
    jitter_frac::Float64 = 0.05,
)
    # 默认值基线
    Ts, target, err = load_sT3_csv(csv_path)
    ints = get_nodes(p_num)
    base_loss, _ = sT3_fit_objective([a1_0, a2_0], Ts, target, err, ints, x0_init)
    best = (a1 = a1_0, a2 = a2_0, loss = Inf)
    best_res = run_train(
        csv_path = csv_path,
        x0_init = x0_init,
        a1_0 = a1_0,
        a2_0 = a2_0,
        maxiter = maxiter,
        lr = lr,
        p_num = p_num,
        grad_eps = grad_eps
    )

    best = (a1 = best_res.a1, a2 = best_res.a2, loss = best_res.loss)

    local_rng = Random.RandomDevice()
    for k in 1:n_restarts
        seed = rand(local_rng, 1:typemax(UInt16))
        rng = Random.MersenneTwister(seed)
        a1k = a1_0 * (1 + jitter_frac * (2*rand(rng) - 1))
        a2k = max(1e-4, a2_0 * (1 + jitter_frac * (2*rand(rng) - 1)))

        res = run_train(
            csv_path = csv_path,
            x0_init = x0_init,
            a1_0 = a1k,
            a2_0 = a2k,
            maxiter = maxiter,
            lr = lr,
            p_num = p_num,
            grad_eps = grad_eps
        )

        if res.loss < best.loss
            best = (a1 = res.a1, a2 = res.a2, loss = res.loss)
            best_res = res
        end
        println("reseed=$k start=(a1=$a1k, a2=$a2k) -> loss=$(Float64(res.loss))")
    end

    return best_res, best
end

"""
    solve_Tmu_precise_130(;
        csv_path = "sT3.csv",
        x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1],
        a1_0::Float64 = -9.8,
        a2_0::Float64 = 0.26,
        p_num::Int = 200,
        jitter::Float64 = 0.12,
        n_restarts::Int = 40,
    )

    在第一个温度点(默认130MeV)上用多初值重启，寻找更稳健的初解。
"""
function solve_Tmu_precise_130(;
    csv_path::String = joinpath(@__DIR__, "sT3.csv"),
    x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1],
    a1_0::Float64 = -9.8,
    a2_0::Float64 = 0.26,
    p_num::Int = 200,
    jitter::Float64 = 0.12,
    n_restarts::Int = 40,
)
    Ts, _, _ = load_sT3_csv(csv_path)
    if isempty(Ts)
        error("sT3.csv 为空，无法确定 130 MeV 初值。")
    end

    if Ts[1] != 130.0
        println(">> 注意：当前sT3首个温度为 ", Ts[1], "MeV，而不是 130 MeV")
    end

    ints = get_nodes(p_num)
    T130 = Ts[1] / hc
    base = map(Float64, x0_init)

    mu_B = zero(T130)

    function try_one(x_try)
        try
            fWrapper(Xs) = Quark_mu_param(Xs, mu_B, T130, ints, a1_0, a2_0)
            x_ok = nonlinear_zero(fWrapper, x_try; ftol = 1e-14, xtol = 1e-14, iterations = 800)
            if !all(isfinite, x_ok)
                return nothing, Inf
            end
            r = fWrapper(x_ok)
            return x_ok, norm(r)
        catch
            return nothing, Inf
        end
    end

    best_x = base
    best_norm = Inf
    n_try = 0

    # 固定确定性网格初值，避免随机性影响可复现
    deltas = [0.0, -0.08, 0.08, -0.15, 0.15, -0.25, 0.25]
    candidates = Vector{Vector{Float64}}()
    push!(candidates, copy(base))

    for d in deltas
        x = copy(base)
        x .+= d .* (abs.(x) .+ 0.2)
        push!(candidates, x)
    end

    # 按单点方向扫动，形成局部重启组
    for j in eachindex(base)
        for d in deltas
            x = copy(base)
            x[j] += d * (abs(base[j]) + 0.2)
            push!(candidates, x)
        end
    end

    rng = MersenneTwister(20260623)
    for k in 1:n_restarts
        x = copy(base)
        x .*= (1 .+ jitter .* (2 .* rand(rng, length(x)) .- 1))
        push!(candidates, x)
    end

    for cand in candidates
        x_try = map(Float64, cand)
        x_candidate, nrm = try_one(x_try)
        n_try += 1
        if x_candidate !== nothing && nrm < best_norm
            best_norm = nrm
            best_x = x_candidate
        end
    end

    println(">> 130MeV 重启求解尝试次数: ", n_try, ", 最优残差 = ", best_norm)
    if isfinite(best_norm) && all(isfinite, best_x)
        println(">> 130MeV 精确初解 = ", best_x)
    end
    if !isfinite(best_norm) || any(!isfinite, best_x)
        println(">> 警告：未获得可靠 130MeV 初解，退回原始初值。")
        return base
    end
    return best_x
end


"""
    run_baseline(;
        csv_path = "sT3.csv",
        x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1],
        a1_0 = -9.8,
        a2_0 = 0.26,
        p_num = 200,
    )

    先在当前参数(a1_0, a2_0)下计算基线 loss。
"""
function run_baseline(;
    csv_path::String = joinpath(@__DIR__, "sT3.csv"),
    x0_init = [-1.8, -1.8, -2.2, 0.1, 0.1],
    a1_0::Float64 = -9.8,
    a2_0::Float64 = 0.26,
    p_num::Int = 200,
)
    Ts, target, err = load_sT3_csv(csv_path)
    ints = get_nodes(p_num)
    baseline_loss, _ = sT3_fit_objective([a1_0, a2_0], Ts, target, err, ints, x0_init)
    return baseline_loss
end

"""
    save_fit_csv(path, res)

    保存拟合结果 (T, sT3_target, sT3_pred, err) 到 csv。
"""
function save_fit_csv(path::String, res)
    Ts = res.Ts
    data = hcat(Ts, res.target, res.pred, res.err)
    header = ["T", "sT3_target", "sT3_pred", "err"]
    open(path, "w") do io
        println(io, join(header, ","))
        for i in 1:size(data, 1)
            @printf(io, "%g,%g,%g,%g\n", data[i, 1], data[i, 2], data[i, 3], data[i, 4])
        end
    end
end

function main()
    println(">> 方法说明：")
    println("   1) 使用 ForwardDiff 计算目标函数关于 (a1, a2) 的梯度（全量批次）")
    println("   2) 用固定步长的梯度下降更新 θ=[a1, a2]")
    println("   3) 每个温度点用 warm-start 的解作为初值，减少 NonlinearSolve 的重复收敛成本")

    x0_130 = solve_Tmu_precise_130()
    baseline = run_baseline(x0_init = x0_130)
    println(">> 基线损失 (当前参数 a1=const a2=const): ", Float64(baseline))

    res = run_train(x0_init = x0_130)
    res_local, best_summary = perturb_and_train(x0_init = x0_130)
    if best_summary.loss < res.loss
        res = res_local
    end

    println(">> 拟合完成")
    @printf("   a1 = %.10f\n", res.a1)
    @printf("   a2 = %.10f\n", res.a2)
    @printf("   loss = %.10f\n", Float64(res.loss))

    out_path = joinpath(@__DIR__, "sT3_fit_result.csv")
    save_fit_csv(out_path, res)
    println(">> 输出预测数据: ", out_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
