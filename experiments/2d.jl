using Pkg
Pkg.activate(@__DIR__)

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
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_samples=100);
# densities1 = first_layer_mixtures(csbm; max_neighbors=50);
densities = [empirical_mixtures(rng, csbm; nb_layers=l, nb_samples=2) for l in 0:L];

plot_2d(csbm, histograms, densities)

mix0 = Mixture(densities0, sbm.S ./ sum(sbm.S))
mix1 = Mixture(densities1, sbm.S ./ sum(sbm.S))

error_montecarlo(rng, mix0; nb_dist_samples=100, nb_error_samples=100)
error_montecarlo(rng, mix1; nb_dist_samples=100, nb_error_samples=100)

error_quadrature_2d(mix0)
error_quadrature_2d(mix1)
