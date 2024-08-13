using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using DensityInterface
using LaTeXStrings
using LinearAlgebra
using MonteCarloMeasurements
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## 1D

sbm = SBM(300, 3, 0.02, 0.01)

features = [
    UnivariateNormal(-1.0, 0.03), UnivariateNormal(-0.0, 0.01), UnivariateNormal(+1.0, 0.02)
]

csbm = CSBM(sbm, features)

L = 4
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_graphs=1000);
# densities1 = first_layer_densities(csbm; max_neighbors=50);
densities = @time random_walk_densities(rng, csbm; nb_layers=L, nb_graphs=100);

plot_1d(csbm, histograms, densities)

errors = error_by_depth(
    rng, csbm; nb_layers=10, nb_trajectories=2, nb_graphs=100, nb_samples=100
)

## 2D

sbm = SBM(300, 3, 0.03, 0.01)

features = [
    BivariateNormal([-1.0, 1.0], [1, 0.1, 0.1, 0.5]),  #
    BivariateNormal([1.0, -1.0], [0.5, 0.2, 0.2, 1.0]),  #
    BivariateNormal([1.0, 1.0], [1, 0.3, 0.3, 1]),  #
]

csbm = CSBM(sbm, features)

L = 3
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_graphs=100);
# densities1 = first_layer_densities(csbm; max_neighbors=50);
densities = random_walk_densities(rng, csbm; nb_layers=L, nb_graphs=20);

plot_2d(csbm, histograms, densities)

errors = error_by_depth(
    rng, csbm; nb_layers=10, nb_trajectories=5, nb_graphs=100, nb_samples=100
)

optimal_depth(rng, csbm; nb_layers=2, nb_trajectories=10, nb_graphs=10, nb_samples=10)
