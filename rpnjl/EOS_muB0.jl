#!/usr/bin/env julia

using Printf
using DelimitedFiles
using LinearAlgebra
using ForwardDiff
using Plots
using Plots.PlotMeasures

include("rpnjl.jl")

const POLY_A0 = 6.75
const POLY_B3 = 0.805
const POLY_B4 = 7.555

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

function load_hrg_table(path::AbstractString)
    raw = readdlm(path, skipstart = 1)
    return (
        T = Float64.(raw[:, 1]),
        pT4 = Float64.(raw[:, 2]),
        eT4 = Float64.(raw[:, 3]),
        sT3 = Float64.(raw[:, 4]),
        IT4 = Float64.(raw[:, 5]),
        cs2 = Float64.(raw[:, 7]),
        CVT3 = Float64.(raw[:, 8]),
        nBT3 = Float64.(raw[:, 14]),
    )
end

function load_lqcd_table(path::AbstractString)
    raw = readdlm(path, ',', skipstart = 1)
    return (
        T = Float64.(raw[:, 1]),
        IT4 = Float64.(raw[:, 2]),
        IT4_err = Float64.(raw[:, 3]),
        pT4 = Float64.(raw[:, 4]),
        pT4_err = Float64.(raw[:, 5]),
        eT4 = Float64.(raw[:, 6]),
        eT4_err = Float64.(raw[:, 7]),
        sT3 = Float64.(raw[:, 8]),
        sT3_err = Float64.(raw[:, 9]),
        CVT3 = Float64.(raw[:, 10]),
        CVT3_err = Float64.(raw[:, 11]),
        cs2 = Float64.(raw[:, 12]),
        cs2_err = Float64.(raw[:, 13]),
    )
end

function default_hrg_path()
    return joinpath(dirname(@__DIR__), "fit_data", "HRG.txt")
end

function default_lqcd_path()
    return joinpath(dirname(@__DIR__), "fit_data", "hotqcd_1407_6387_table1_eos_origin_yerr.csv")
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
        total += calculate_vacuum_term(p1, w1, masses[flavor])
        total += calculate_thermal_term(p2, w2, masses[flavor], T, mu, Phi1, Phi2)
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
    fvec = zeros(Tout, 5)
    fvec[1:5] = dOmega_dorder_poly_T0(X0[1:5], mu_B, T, ints, theta)
    return fvec
end

function solve_order_poly_T0(T, ints, x0::AbstractVector, theta::AbstractVector)
    fWrapper(Xs) = quark_mu_poly_T0(Xs, zero(T), T, ints, theta)
    x = nonlinear_zero(fWrapper, x0; ftol = 1e-11, xtol = 1e-11, iterations = 600)
    rnorm = norm(fWrapper(x))
    if !all(isfinite, x) || !isfinite(rnorm)
        error("invalid order-parameter solution")
    end
    return Float64.(x), rnorm
end

function solve_order_best_poly_T0(T, ints, theta::AbstractVector)
    candidates = [
        [-1.88, -1.88, -2.60, 0.001, 0.001],
        [-1.8, -1.8, -2.2, 0.1, 0.1],
        [-1.9, -1.9, -2.7, 0.02, 0.02],
        [-1.7, -1.7, -2.4, 0.05, 0.05],
    ]
    best_x = copy(candidates[1])
    best_norm = Inf
    for cand in candidates
        try
            x, rnorm = solve_order_poly_T0(T, ints, cand, theta)
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

function cV_ad_implicit_poly_T0(X::AbstractVector, T::Float64, ints, theta::AbstractVector)
    z0 = vcat(Float64.(X), T)
    H = ForwardDiff.hessian(z -> Omega_poly_T0(z[1:5], 0.0, z[6], ints, theta), z0)
    Hxx = Symmetric(Matrix{Float64}(H[1:5, 1:5]))
    HxT = Vector{Float64}(H[1:5, 6])
    HTT = Float64(H[6, 6])
    P_TT = -HTT + dot(HxT, Hxx \ HxT)
    return T * P_TT
end

function baryon_density_T3(X::AbstractVector, T::Float64, ints, theta::AbstractVector)
    dOmega_dmuB = ForwardDiff.derivative(muB -> begin
        Omega_poly_T0(X, muB, T, ints, theta)
    end, 0.0)
    return -dOmega_dmuB / T^3
end

function compute_eos(theta::AbstractVector, Ts_mev::Vector{Float64}, ints, hrg)
    Tref_mev = 50.0
    Tref = Tref_mev / hc
    href_idx = argmin(abs.(hrg.T .- Tref_mev))
    p_ref = hrg.pT4[href_idx] * Tref^4

    Xref, rref = solve_order_best_poly_T0(Tref, ints, theta)
    omega_ref = Float64(Omega_poly_T0(Xref, 0.0, Tref, ints, theta))
    pressure_shift = p_ref + omega_ref

    n = length(Ts_mev)
    omega = Vector{Float64}(undef, n)
    pT4 = Vector{Float64}(undef, n)
    sT3 = Vector{Float64}(undef, n)
    eT4 = Vector{Float64}(undef, n)
    IT4 = Vector{Float64}(undef, n)
    nBT3 = Vector{Float64}(undef, n)
    CVT3 = Vector{Float64}(undef, n)
    cs2 = Vector{Float64}(undef, n)
    residual = Vector{Float64}(undef, n)
    phi_u = Vector{Float64}(undef, n)
    phi_d = Vector{Float64}(undef, n)
    phi_s = Vector{Float64}(undef, n)
    Phi = Vector{Float64}(undef, n)
    PhiBar = Vector{Float64}(undef, n)

    X = Xref
    for i in eachindex(Ts_mev)
        T = Ts_mev[i] / hc
        if i == 1
            X, residual[i] = solve_order_best_poly_T0(T, ints, theta)
        else
            X, residual[i] = solve_order_poly_T0(T, ints, X, theta)
        end

        omega[i] = Float64(Omega_poly_T0(X, 0.0, T, ints, theta))
        p = -omega[i] + pressure_shift
        s = Float64(-dOmega_dT_poly_T0(X, 0.0, T, ints, theta))
        cv = cV_ad_implicit_poly_T0(X, T, ints, theta)

        pT4[i] = p / T^4
        sT3[i] = s / T^3
        eT4[i] = sT3[i] - pT4[i]
        IT4[i] = sT3[i] - 4 * pT4[i]
        nBT3[i] = baryon_density_T3(X, T, ints, theta)
        CVT3[i] = cv / T^3
        cs2[i] = s / cv
        phi_u[i], phi_d[i], phi_s[i], Phi[i], PhiBar[i] = X
    end

    return (
        T = Ts_mev,
        pT4 = pT4,
        IT4 = IT4,
        sT3 = sT3,
        eT4 = eT4,
        nBT3 = nBT3,
        CVT3 = CVT3,
        cs2 = cs2,
        phi_u = phi_u,
        phi_d = phi_d,
        phi_s = phi_s,
        Phi = Phi,
        PhiBar = PhiBar,
        residual = residual,
        pressure_shift = pressure_shift,
        pressure_ref_T_MeV = Tref_mev,
        pressure_ref_pT4 = hrg.pT4[href_idx],
        pressure_ref_residual = rref,
    )
end

function save_eos_csv(path::AbstractString, eos, theta::AbstractVector)
    open(path, "w") do io
        println(io, "T_MeV,muB_MeV,P_over_T4,I_over_T4,s_over_T3,e_over_T4,nB_over_T3,CV_over_T3,cs2,phi_u,phi_d,phi_s,Phi,PhiBar,solve_residual,a1,a2,T0_MeV")
        for i in eachindex(eos.T)
            @printf(io, "%.12g,0,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                eos.T[i], eos.pT4[i], eos.IT4[i], eos.sT3[i], eos.eT4[i], eos.nBT3[i],
                eos.CVT3[i], eos.cs2[i], eos.phi_u[i], eos.phi_d[i], eos.phi_s[i],
                eos.Phi[i], eos.PhiBar[i], eos.residual[i], theta[1], theta[2], theta[3])
        end
    end
end

function save_eos_plot(path::AbstractString, eos, hrg, lqcd, theta::AbstractVector)
    xlim = (minimum(eos.T), maximum(eos.T))
    title_text = @sprintf("a1=%.6f, a2=%.6f, T0=%.3f MeV", theta[1], theta[2], theta[3])
    plot_kwargs = (
        left_margin = 20mm,
        right_margin = 5mm,
        bottom_margin = 7mm,
        top_margin = 5mm,
        guidefontsize = 11,
        tickfontsize = 9,
        legendfontsize = 8,
    )

    p1 = plot(eos.T, eos.pT4, lw = 2, label = "RPNJL", xlabel = "T (MeV)", ylabel = "P/T^4", title = title_text; plot_kwargs...)
    plot!(p1, hrg.T, hrg.pT4, ls = :dash, lw = 1.8, label = "HRG")
    scatter!(p1, lqcd.T, lqcd.pT4, yerror = lqcd.pT4_err, ms = 2.5, label = "LQCD")

    p2 = plot(eos.T, eos.IT4, lw = 2, label = "RPNJL", xlabel = "T (MeV)", ylabel = "I/T^4"; plot_kwargs...)
    plot!(p2, hrg.T, hrg.IT4, ls = :dash, lw = 1.8, label = "HRG")
    scatter!(p2, lqcd.T, lqcd.IT4, yerror = lqcd.IT4_err, ms = 2.5, label = "LQCD")

    p3 = plot(eos.T, eos.sT3, lw = 2, label = "RPNJL", xlabel = "T (MeV)", ylabel = "s/T^3"; plot_kwargs...)
    plot!(p3, hrg.T, hrg.sT3, ls = :dash, lw = 1.8, label = "HRG")
    scatter!(p3, lqcd.T, lqcd.sT3, yerror = lqcd.sT3_err, ms = 2.5, label = "LQCD")

    p4 = plot(eos.T, eos.nBT3, lw = 2, label = "RPNJL", xlabel = "T (MeV)", ylabel = "n_B/T^3"; plot_kwargs...)
    plot!(p4, hrg.T, hrg.nBT3, ls = :dash, lw = 1.8, label = "HRG")
    hline!(p4, [0.0], color = :gray, ls = :dot, label = "")

    p5 = plot(eos.T, eos.CVT3, lw = 2, label = "RPNJL", xlabel = "T (MeV)", ylabel = "C_V/T^3"; plot_kwargs...)
    plot!(p5, hrg.T, hrg.CVT3, ls = :dash, lw = 1.8, label = "HRG")
    scatter!(p5, lqcd.T, lqcd.CVT3, yerror = lqcd.CVT3_err, ms = 2.5, label = "LQCD")

    p6 = plot(eos.T, eos.cs2, lw = 2, label = "RPNJL", xlabel = "T (MeV)", ylabel = "c_s^2", ylim = (0, 0.45); plot_kwargs...)
    plot!(p6, hrg.T, hrg.cs2, ls = :dash, lw = 1.8, label = "HRG")
    scatter!(p6, lqcd.T, lqcd.cs2, yerror = lqcd.cs2_err, ms = 2.5, label = "LQCD")

    plt = plot(
        p1, p2, p3, p4, p5, p6,
        layout = (3, 2),
        size = (1700, 1250),
        xlim = xlim,
        left_margin = 24mm,
    )
    savefig(plt, path)
end

function run()
    a1p = parse_float_arg("--a1", -7.505272993997)
    a2p = parse_float_arg("--a2", 0.091848942829)
    T0_mev = parse_float_arg("--T0", 175.695929621953)
    p_num = parse_int_arg("--p_num", 80)
    Tmin = parse_float_arg("--Tmin", 50.0)
    Tmax = parse_float_arg("--Tmax", 400.0)
    Tstep = parse_float_arg("--Tstep", 2.0)

    theta = [a1p, a2p, T0_mev]
    ints = get_nodes(p_num)
    hrg = load_hrg_table(default_hrg_path())
    lqcd = load_lqcd_table(default_lqcd_path())
    Ts = range_values(Tmin, Tmax, Tstep)

    println("EOS at muB=0 for polynomial Polyakov potential")
    @printf("a1=%.12f, a2=%.12f, T0=%.12f MeV, p_num=%d\n", theta[1], theta[2], theta[3], p_num)
    println("Pressure normalization: P(T=50 MeV, muB=0) = P_HRG(T=50 MeV)")

    eos = compute_eos(theta, Ts, ints, hrg)

    out_dir = joinpath(@__DIR__, "data", "eos_muB0_poly")
    mkpath(out_dir)
    tag = @sprintf("eos_muB0_poly_a1_%s_a2_%s_T0_%s", tag_param(theta[1]), tag_param(theta[2]), tag_param(theta[3]))
    csv_path = joinpath(out_dir, "$(tag).csv")
    svg_path = joinpath(out_dir, "$(tag).svg")

    save_eos_csv(csv_path, eos, theta)
    save_eos_plot(svg_path, eos, hrg, lqcd, theta)

    @printf("pressure reference: T=%.3f MeV, P_HRG/T^4=%.12g, solve residual=%.3e\n",
        eos.pressure_ref_T_MeV, eos.pressure_ref_pT4, eos.pressure_ref_residual)
    println(">> saved EOS csv: ", csv_path)
    println(">> saved EOS svg: ", svg_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run()
end
