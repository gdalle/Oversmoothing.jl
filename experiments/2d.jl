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

sbm = SBM(90, 3, 0.03, 0.01)

features = [
    MultivariateNormal(SVector(-1.0, 1.0), SMatrix{2,2}(0.3, 0.1, 0.1, 0.3)),  #
    MultivariateNormal(SVector(1.0, -1.0), SMatrix{2,2}(0.1, 0, 0, 0.5)),  #
    MultivariateNormal(SVector(1.0, 1.0), SMatrix{2,2}(0.2, -0.1, -0.1, 0.2)),  #
]

csbm = CSBM(sbm, features)

## Computation

histograms0, histograms1 = @time embeddings(
    rng, csbm; nb_layers=1, resample_graph=true, nb_samples=100
);
densities0 = [Mixture([features[c]], [1.0]) for c in eachindex(features)];
densities1 = first_layer_mixtures(csbm; max_neighbors=50);

plot_2d(csbm, histograms0, densities0; layer=0)
plot_2d(csbm, histograms1, densities1; layer=1)

mix0 = Mixture(densities0, sbm.S ./ sum(sbm.S))
mix1 = Mixture(densities1, sbm.S ./ sum(sbm.S))

error_montecarlo(rng, mix0; nb_dist_samples=100, nb_error_samples=100)
error_montecarlo(rng, mix1; nb_dist_samples=100, nb_error_samples=100)

error_quadrature_2d(mix0)
error_quadrature_2d(mix1)
