using Pkg
Pkg.activate(dirname(@__DIR__))

using DensityInterface
using LinearAlgebra
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## Instance

sbm = SBM(300, 3, 0.03, 0.01)

features = [
    MultivariateNormal(SVector(-1.0, 1.0), SMatrix{2,2}(1, 0.1, 0.1, 0.5)),  #
    MultivariateNormal(SVector(1.0, -1.0), SMatrix{2,2}(0.5, 0.2, 0.2, 1.0)),  #
    MultivariateNormal(SVector(1.0, 1.0), SMatrix{2,2}(1, 0.3, 0.3, 1)),  #
]

csbm = CSBM(sbm, features)

## Computation

L = 3
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_graphs=100);
# densities1 = first_layer_mixtures(csbm; max_neighbors=50);
densities = random_walk_mixtures(rng, csbm; nb_layers=L, nb_graphs=20);

plot_2d(csbm, histograms, densities)

errors = random_walk_errors(rng, csbm; nb_layers=10, nb_graphs=100, nb_samples=100)
