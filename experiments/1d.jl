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

sbm = SBM(3000, 3, 0.002, 0.001)

features = [
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(0.03)),  #
    MultivariateNormal(SVector(0.0), SMatrix{1,1}(0.01)),  #
    MultivariateNormal(SVector(+1.0), SMatrix{1,1}(0.02)),  #
]

csbm = CSBM(sbm, features)

## Computation

L = 3
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_samples=100);
# densities1 = first_layer_mixtures(csbm; max_neighbors=50);
densities = [empirical_mixtures(rng, csbm; nb_layers=l, nb_samples=10) for l in 0:L];

plot_1d(csbm, histograms, densities)

error_montecarlo(rng, mix0; nb_dist_samples=100, nb_error_samples=100)
error_montecarlo(rng, mix0; nb_dist_samples=100, nb_error_samples=100)

error_quadrature_1d(mix0)
error_quadrature_1d(mix1)
