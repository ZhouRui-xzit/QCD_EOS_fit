using ForwardDiff
using DataFrames
using CSV

include("constants.jl")
include("rpnjl.jl")

let 
    ints = get_nodes(200)

    X0 = [-1.8, -1.8, -2.2, 0.1, 0.1]

    Ts = 50.0:2.0:400.0
    muB = 0.0


    EOS_s = NamedTuple[]
    for (i, T) in enumerate(Ts)
        X0 = Tmu(X0, muB / hc, T / hc, ints)

        s = -dOmega_dT(X0, [muB / hc, muB / hc, muB / hc], T / hc, ints)
        push!(EOS_s, (T = T, muB = muB, s = s/(T/hc)^3))
        if i % 5 == 0
            println("T = $T MeV, muB = $muB MeV, X = $(X0), s = $(s/(T/hc)^3)")
        end
    end

    df = DataFrame(EOS_s)

    CSV.write("EOS_muB0.csv", df)

    println(">> 已保存 EOS_muB0.csv")
end