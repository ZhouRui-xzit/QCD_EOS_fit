include("constants.jl")


using ForwardDiff # AD
using NaNMath   # nanlog
using FastGaussQuadrature  # Gauss-Legendre 积分
import NonlinearSolve
using DelimitedFiles
using LinearAlgebra

#  gauss 节点 
function gauleg(a, b, n)
    x, w = gausslegendre(n)
    x_mapped = (b - a) / 2 .* x .+ (b + a) / 2
    w_mapped = (b - a) / 2 .* w
    return x_mapped, w_mapped
end

function get_nodes(p_num)
    p1, w1 = gauleg(0.0, Lambda, p_num)
    p2, w2 = gauleg(0.0, 30.0, p_num)
    w1 = w1 .* p1.^2 ./ (2*pi^2)
    w2 = w2 .* p2.^2 ./ (2*pi^2)
    int1 = (p1, w1)
    int2 = (p2, w2)
    return int1, int2
end

function solve_nonlinear_system(f, x0::AbstractVector; ftol = 1e-8, xtol = 1e-8, iterations = 1000)
    prob = NonlinearSolve.NonlinearProblem((u, p) -> f(u), copy(x0))
    sol = NonlinearSolve.solve(prob; abstol = ftol, reltol = xtol, maxiters = iterations)
    return sol.u, sol
end

function nonlinear_zero(f, x0::AbstractVector; kwargs...)
    x, _ = solve_nonlinear_system(f, x0; kwargs...)
    return x
end





function Omega(orders, mus, T, ints)

    p1, w1 = ints[1]
    p2, w2 = ints[2]

    phi = orders[1:3]
    Phi1 = orders[4]
    Phi2 = orders[5]
    chi = chiral(phi)
    U = calc_U(T, Phi1, Phi2)
    Masses = Mass(phi)
    Omega_q = 0.0
    for flavor = 1:3
        mu = mus[flavor] 
        mass = Masses[flavor]
        vacuum_term =  calculate_vacuum_term(p1, w1, mass)
        thermal_term = calculate_thermal_term(p2, w2, mass, T, mu, Phi1, Phi2)
        Omega_q += vacuum_term + thermal_term
    end

    Omega_total = chi + U + Omega_q 
    return Omega_total      
end


function calculate_vacuum_term(p, w, mass)
    E = sqrt.(p.^2 .+ mass^2)
    integrand = w.* E  # 积分测度包含于w中
    return -2 * Nc * sum(integrand)
end

function calculate_thermal_term(p, w, mass, T, mu, Phi1, Phi2)
    E = sqrt.(p.^2 .+ mass^2)
    E_minus = E .- mu
    E_plus = E .+ mu

    log_sum = log.(AA(E_minus, T, Phi1, Phi2)) .+ log.(AAbar(E_plus, T, Phi1, Phi2))
    integrand = w .* log_sum  # 积分测度包含于w中
    return -2*T * sum(integrand)
end



# 对序参量求导
function dOmega_dorder(orders, mus, T, ints)
    return ForwardDiff.gradient(x -> Omega(x, mus, T, ints), orders)
end
 


# 对mu求导
function dOmega_dmus(orders, mus, T, ints)
    return ForwardDiff.gradient(x -> Omega(orders, x, T, ints), mus)
end

# 对T求导
function dOmega_dT(orders, mus, T, ints)
    return ForwardDiff.derivative(x -> Omega(orders, mus, x, ints), T)
end

# TT derv
function dOmega_dT_T(orders, mus, T, ints)
    return ForwardDiff.derivative(x -> dOmega_dT(orders, mus, x, ints), T)
end







function AA(x, T, Phi1, Phi2)
    """
    夸克分布归一化分母（向量化版本）
    
    支持向量输入，计算每个x对应的分母项
    """
    term1 = exp.(-x ./ T)
    term2 = exp.(-2.0 .* x ./ T)
    term3 = exp.(-3.0 .* x ./ T)

    result = 1.0 .+ 3.0 .* Phi1 .* term1 .+ 3.0 .* Phi2 .* term2 .+ term3
    return result
end

function AAbar(x, T, Phi1, Phi2)
    """
    反夸克分布归一化分母（向量化版本）
    
    支持向量输入，计算每个x对应的分母项
    """
    term1 = exp.(-x ./ T)
    term2 = exp.(-2.0 .* x ./ T)
    term3 = exp.(-3.0 .* x ./ T)

    result = 1.0 .+ 3.0 .* Phi2 .* term1 .+ 3.0 .* Phi1 .* term2 .+ term3
    return result
end



function chiral(phi)
    
    term1 = g_S * sum(phi[1:3].^2) #/2
    term2 = -g_D/2 * (phi[1] * phi[2] * phi[3])
    term3 = 3*g1/2 * (sum(phi[1:3].^2))^2
    term4 = 3*g2 * sum(phi[1:3].^4)

    return term1 + term2 + term3 + term4
end




function Mass(phi)
    """计算三种夸克的有效质量"""
    mass_u = mass_u0 - 2*g_S*phi[1]+g_D/4*phi[2]*phi[3]-2*g1*phi[1]*(sum(phi[1:3].^2))-4*g2*phi[1]^3
    mass_d = mass_d0 - 2*g_S*phi[2]+g_D/4*phi[1]*phi[3]-2*g1*phi[2]*(sum(phi[1:3].^2))-4*g2*phi[2]^3
    mass_s = mass_s0 - 2*g_S*phi[3]+g_D/4*phi[1]*phi[2]-2*g1*phi[3]*(sum(phi[1:3].^2))-4*g2*phi[3]^3
    #mass_u = mass_u0 - 2*g_S*phi[1]+g_D*phi[2]*phi[3]-2*g1*phi[1]*(sum(phi[1:3].^2))-4*g2*phi[1]^3
    #mass_d = mass_d0 - 2*g_S*phi[2]+g_D*phi[1]*phi[3]-2*g1*phi[2]*(sum(phi[1:3].^2))-4*g2*phi[2]^3
    #mass_s = mass_s0 - 2*g_S*phi[3]+g_D*phi[1]*phi[2]-2*g1*phi[3]*(sum(phi[1:3].^2))-4*g2*phi[3]^3



    return [mass_u, mass_d, mass_s]
end


function calc_U(T, Phi1, Phi2)
    b2 = a0 + a1 * T0/T * exp(-a2 * T/T0)
    term1 = -b2/2 * Phi1 * Phi2 - b3/6 * (Phi1^3 + Phi2^3) + b4/4 * (Phi1 * Phi2)^2
    J = (27/(24*pi^2)) * (1 - 6*(Phi1*Phi2) + 4*(Phi1^3 + Phi2^3) - 3*(Phi1*Phi2)^2)
    return T^4*term1
end







"""
    计算给定T和mu_B下的序参量
    X0 = [phi_u, phi_d, phi_s, Phi1, Phi2] 初始猜测值
"""

function Quark_mu(X0, mu_B, T, ints)

    T_out = promote_type(eltype(X0), typeof(T), typeof(mu_B))
    orders = X0[1:5]
    mus = [1/3*mu_B, 1/3*mu_B, 1/3*mu_B]

    fvec = zeros(T_out, 5)
    fvec[1:5] = dOmega_dorder(orders, mus, T, ints)

    return fvec
end




"""
    计算给定T和rho_B下的序参量
    X0 = [phi_u, phi_d, phi_s, Phi1, Phi2, muB, muQ] 初始猜测值
"""

function Quark_rho(X0, T, rho, ints)
    T_out = promote_type(eltype(X0), typeof(T), typeof(rho))

    fvec = zeros(T_out, 6)
    orders = X0[1:5]
    mu_B = X0[6]
    mus = [1/3*mu_B, 1/3*mu_B, 1/3*mu_B]
    fvec[1:5] = dOmega_dorder(orders, mus, T, ints)
    rho_now = - sum(dOmega_dmus(orders, mus, T, ints)) / rho0
    fvec[6] = rho_now - rho
    return fvec
end


function Tmu(X0, mu_B, T, ints)
    fWrapper(Xs) = Quark_mu(Xs, mu_B, T, ints)
    NewX = nonlinear_zero(fWrapper, X0)
    return NewX
end




function Trho(X0, T, rho, ints)
    fWrapper(Xs) = Quark_rho(Xs, T, rho, ints)
    NewX = nonlinear_zero(fWrapper, X0)
    return NewX
end



"""
    calc_U_param(T, Phi1, Phi2, a1p, a2p)
    参数化 Polyakov 势，允许对 a1/a2 求导和优化
"""
function calc_U_param(T, Phi1, Phi2, a1p, a2p)
    b2 = a0 + a1p * T0/T * exp(-a2p * T/T0)
    term1 = -b2/2 * Phi1 * Phi2 - b3/6 * (Phi1^3 + Phi2^3) + b4/4 * (Phi1 * Phi2)^2
    J = (27/(24*pi^2)) * (1 - 6*(Phi1*Phi2) + 4*(Phi1^3 + Phi2^3) - 3*(Phi1*Phi2)^2)
    return T^4*term1
end


function calc_log_U(T, Phi1, Phi2)
    x=T0 / T
    aT = a0 + a1 * x + a2 * x^2
    log_term = log(1 - 6*(Phi1*Phi2) + 4*(Phi1^3 + Phi2^3) - 3*(Phi1*Phi2)^2)
    term = -aT/2 * Phi1 * Phi2 + b3 * x^3 * log_term
    return T^4*term
end








"""
    Omega_param(orders, mus, T, ints, a1p, a2p)
"""
function Omega_param(orders, mus, T, ints, a1p, a2p)
    p1, w1 = ints[1]
    p2, w2 = ints[2]

    phi = orders[1:3]
    Phi1 = orders[4]
    Phi2 = orders[5]
    chi = chiral(phi)
    U = calc_U_param(T, Phi1, Phi2, a1p, a2p)
    Masses = Mass(phi)
    Omega_q = 0.0
    for flavor = 1:3
        mu = mus[flavor] 
        mass = Masses[flavor]
        vacuum_term =  calculate_vacuum_term(p1, w1, mass)
        thermal_term = calculate_thermal_term(p2, w2, mass, T, mu, Phi1, Phi2)
        Omega_q += vacuum_term + thermal_term
    end

    Omega_total = chi + U + Omega_q 
    return Omega_total      
end

"""
    dOmega_dorder_param(orders, mus, T, ints, a1p, a2p)
"""
function dOmega_dorder_param(orders, mus, T, ints, a1p, a2p)
    return ForwardDiff.gradient(x -> Omega_param(x, mus, T, ints, a1p, a2p), orders)
end

"""
    dOmega_dT_param(orders, mus, T, ints, a1p, a2p)
"""
function dOmega_dT_param(orders, mus, T, ints, a1p, a2p)
    return ForwardDiff.derivative(x -> Omega_param(orders, mus, x, ints, a1p, a2p), T)
end

"""
    Quark_mu_param(X0, mu_B, T, ints, a1p, a2p)
    给定 a1/a2 时的方程残差
"""
function Quark_mu_param(X0, mu_B, T, ints, a1p, a2p)
    T_out = promote_type(eltype(X0), typeof(T), typeof(mu_B), typeof(a1p), typeof(a2p))
    orders = X0[1:5]
    mus = [1/3*mu_B, 1/3*mu_B, 1/3*mu_B]

    fvec = zeros(T_out, 5)
    fvec[1:5] = dOmega_dorder_param(orders, mus, T, ints, a1p, a2p)

    return fvec
end

"""
    Tmu_param(X0, mu_B, T, ints, a1p, a2p)
    a1/a2 下的 Tmu 求解
"""
function Tmu_param(X0, mu_B, T, ints, a1p, a2p)
    fWrapper(Xs) = Quark_mu_param(Xs, mu_B, T, ints, a1p, a2p)
    NewX = nonlinear_zero(fWrapper, X0)
    return NewX
end

"""
    load_sT3_csv(file_path)
    读取 sT3.csv，返回 T/sT3/err 列
"""
function load_sT3_csv(file_path::AbstractString)
    data = readdlm(file_path, ',')
    if size(data, 1) < 2 || size(data, 2) < 2
        error("非法的 sT3 文件：$file_path")
    end

    Ts = Float64.(data[2:end, 1])
    target = Float64.(data[2:end, 2])
    err = Float64.(data[2:end, min(size(data, 2), 5)])
    return Ts, target, err
end

"""
    sT3_fit_objective(theta, Ts, target, err, ints, x0_init)
    给定 (a1, a2) 计算 chi2 和对应模型值
"""
function sT3_fit_objective(theta::AbstractVector, Ts, target, err, ints, x0_init)
    a1p = theta[1]
    a2p = theta[2]
    T_out = promote_type(eltype(theta), eltype(Ts), eltype(target), eltype(err))

    ndata = length(Ts)
    pred = Vector{T_out}(undef, ndata)
    X = map(v -> T_out(v), x0_init)
    loss = zero(T_out)

    for i in 1:ndata
        Ti = Ts[i] / hc
        X = Tmu_param(X, zero(T_out), Ti, ints, a1p, a2p)
        s = -dOmega_dT_param(X, [zero(T_out), zero(T_out), zero(T_out)], Ti, ints, a1p, a2p)
        pred[i] = s / (Ti^3)

        sigma = err[i] > 0 ? T_out(err[i]) : one(T_out)
        diff = (pred[i] - T_out(target[i])) / sigma
        loss += diff * diff
    end

    return loss / 2, pred
end

"""
    train_a1_a2(...)
    利用 forward-mode AD 优化 a1/a2，最小化 sT3 的加权 χ²
"""
function train_a1_a2(;
    csv_path::AbstractString = "sT3.csv",
    x0_init::AbstractVector = [-1.8, -1.8, -2.2, 0.1, 0.1],
    a1_0::Float64 = -9.8,
    a2_0::Float64 = 0.26,
    maxiter::Int = 60,
    lr::Float64 = 0.1,
    p_num::Int = 200,
    verbose::Bool = true,
    grad_eps::Float64 = 1e-8,
)
    Ts, target, err = load_sT3_csv(csv_path)
    ints = get_nodes(p_num)

    θ = Float64[a1_0, a2_0]
    best_θ = copy(θ)
    best_loss = Inf

    for it in 1:maxiter
        function obj_local(v)
            loss, _ = sT3_fit_objective(v, Ts, target, err, ints, x0_init)
            return loss
        end

        g = ForwardDiff.gradient(obj_local, θ)
        θ .-= lr .* g

        # 限定范围，避免进入不稳定区
        θ[1] = clamp(θ[1], -80.0, 0.0)
        θ[2] = max(θ[2], 1e-6)

        cur_loss, pred = sT3_fit_objective(θ, Ts, target, err, ints, x0_init)
        if cur_loss < best_loss
            best_loss = cur_loss
            best_θ = copy(θ)
        end

        if verbose && (it == 1 || it % 10 == 0 || it == maxiter)
            println("iter=$it loss=$(Float64(cur_loss)) a1=$(θ[1]) a2=$(θ[2])")
        end
        if norm(g) < grad_eps
            verbose && println(">>> 梯度范数 < 阈值，提前结束。")
            break
        end
    end

    final_loss, pred = sT3_fit_objective(best_θ, Ts, target, err, ints, x0_init)
    return (
        a1 = best_θ[1],
        a2 = best_θ[2],
        loss = final_loss,
        Ts = Ts,
        pred = pred,
        target = target,
        err = err,
        p_num = p_num,
    )
end


