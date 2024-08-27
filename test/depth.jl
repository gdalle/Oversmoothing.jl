using LinearAlgebra
using Oversmoothing
using StableRNGs
using Test

rng = StableRNG(63)

csbm = CircularCSBM2d(; N=300, C=3, p_in=0.1, p_out=0.02, σ=1.0)

accuracies_th = accuracy_by_depth(
    rng, csbm; method=:randomwalk, nb_layers=5, nb_trajectories=10, nb_graphs=10
);

accuracies = accuracy_by_depth(
    rng, csbm; method=:logisticregression, nb_layers=5, nb_trajectories=10, nb_graphs=100
);

@test value.(accuracies_th) ≈ value.(accuracies) rtol = 1e-2
@test all(<(3e-2), uncertainty.(accuracies_th))
@test all(<(3e-2), uncertainty.(accuracies))

depth = optimal_depth(rng, csbm; nb_layers=5, nb_trajectories=10, nb_graphs=5)
@test uncertainty(depth) < 1
