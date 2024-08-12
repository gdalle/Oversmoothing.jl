using Pkg
Pkg.activate(dirname(@__DIR__))

using CairoMakie
using DensityInterface
using LaTeXStrings
using Latexify
using LinearAlgebra
using MonteCarloMeasurements
using OhMyThreads
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## Instance

sbm = SBM(100, 3, 0.02, 0.01)

features = [
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(0.03)),  #
    MultivariateNormal(SVector(-0.0), SMatrix{1,1}(0.01)),  #
    MultivariateNormal(SVector(+1.0), SMatrix{1,1}(0.02)),  #
]

csbm = CSBM(sbm, features)

## Computation

L = 4
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_graphs=1000);
# densities1 = first_layer_mixtures(csbm; max_neighbors=50);
densities = random_walk_mixtures(rng, csbm; nb_layers=L, nb_graphs=100);

plot_1d(csbm, histograms, densities)

errors = random_walk_errors(rng, csbm; nb_layers=10, nb_graphs=100, nb_samples=100)
