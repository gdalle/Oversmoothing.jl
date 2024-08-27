using LinearAlgebra
using Oversmoothing
using StableRNGs
using Test

rng = StableRNG(63)

csbm = SymmetricCSBM2d(; C=3, din=10, dout=2, σ=1.0)

accuracies_th = accuracy_by_depth(
    rng, csbm, Val(:randomwalk); nb_layers=5, nb_trajectories=10, nb_graphs=10
);

accuracies = accuracy_by_depth(
    rng, csbm, Val(:logisticregression); nb_layers=5, nb_trajectories=10, nb_graphs=100
);

@test value.(accuracies_th) ≈ value.(accuracies) rtol = 1e-2
@test all(<(3e-2), uncertainty.(accuracies_th))
@test all(<(3e-2), uncertainty.(accuracies))
