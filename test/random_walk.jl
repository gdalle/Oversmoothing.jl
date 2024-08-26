using Oversmoothing
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

sbm = SBM(99, 3, 0.002, 0.001)

features = [
    UnivariateNormal(-1.0, 0.03), UnivariateNormal(0.0, 0.01), UnivariateNormal(+1.0, 0.02)
]

csbm = CSBM(sbm, features)

densities = random_walk_densities(rng, csbm; nb_layers=7, nb_graphs=4);
@test size(densities) == (8, 3)

accuracy_trajectories = random_walk_accuracy_trajectories(
    rng, csbm; nb_trajectories=2, nb_layers=7, nb_graphs=4, rtol=1e-3
);
@test size(accuracy_trajectories) == (8, 2)
