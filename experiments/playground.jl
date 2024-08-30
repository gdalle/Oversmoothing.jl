using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using DensityInterface
using LaTeXStrings
using LinearAlgebra
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## 1D

sbm = SBM(300, 3, 0.03, 0.01)
features = [
    UnivariateNormal(-1.0, 0.03), UnivariateNormal(-0.0, 0.01), UnivariateNormal(+1.0, 0.02)
]
csbm = CSBM(sbm, features)

L = 2
embeddings = @time empirical_embeddings(rng, csbm; nb_layers=L, nb_graphs=100);
densities = @time random_walk_densities(rng, csbm; nb_layers=L, nb_graphs=20);

plot_1d(csbm, embeddings, densities; path=joinpath(@__DIR__, "images", "1d.pdf"))

## 2D

sbm = SBM(300, 3, 0.03, 0.01)
features = [
    BivariateNormal([-2.0, 0.0], [1.0 0.0; 0.0 2.0]),
    BivariateNormal([0.0, 2.0], [2.0 -0.4; -0.4 1.0]),
    BivariateNormal([+3.0, -1.0], [1.0 0.3; 0.3 1.0]),
]
csbm = CSBM(sbm, features)

L = 2
embeddings = @time empirical_embeddings(rng, csbm; nb_layers=L, nb_graphs=100);
densities = random_walk_densities(rng, csbm; nb_layers=L, nb_graphs=20);

plot_2d(csbm, embeddings, densities; path=joinpath(@__DIR__, "images", "2d.pdf"))
