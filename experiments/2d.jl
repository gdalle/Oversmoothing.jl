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
    MultivariateNormal(SVector(-1.0, 1.0), SMatrix{2,2}(0.3, 0.1, 0.1, 0.3)),  #
    MultivariateNormal(SVector(1.0, -1.0), SMatrix{2,2}(0.1, 0, 0, 0.5)),  #
    MultivariateNormal(SVector(1.0, 1.0), SMatrix{2,2}(0.2, -0.1, -0.1, 0.2)),  #
]

csbm = CSBM(sbm, features)

## Computation

L = 2
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_graphs=100);
# densities1 = first_layer_mixtures(csbm; max_neighbors=50);
densities = random_walk_mixtures(rng, csbm; nb_layers=L, nb_graphs=2);

plot_2d(csbm, histograms, densities)
