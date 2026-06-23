include(joinpath(@__DIR__, "train_a1_a2.jl"))
res, best = perturb_and_train(
    a1_0 = -9.2,
    a2_0 = 0.215,
    maxiter = 120,
    lr = 0.03,
    p_num = 220,
    grad_eps = 1e-9,
    n_restarts = 20,
    jitter_frac = 0.1,
)
@printf("restarts done\n")
@printf("best a1=%.10f\n", Float64(best.a1))
@printf("best a2=%.10f\n", Float64(best.a2))
@printf("best loss=%.10f\n", Float64(best.loss))
