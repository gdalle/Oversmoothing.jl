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

L = 2

## 1D

csbm = LinearCSBM1d(; N=60, C=3, p_in=0.04, p_out=0.02, σ=0.1)

embeddings = @time empirical_embeddings(rng, csbm; nb_layers=L, nb_graphs=100);
densities = @time random_walk_densities(rng, csbm; nb_layers=L, nb_graphs=20);

plot_1d(csbm, embeddings, densities)

## 2D

csbm = CircularCSBM2d(; N=60, C=3, p_in=0.04, p_out=0.02, σ=0.15, stretch=1)

embeddings = @time empirical_embeddings(rng, csbm; nb_layers=L, nb_graphs=100);
densities = @time random_walk_densities(rng, csbm; nb_layers=L, nb_graphs=20);

plot_2d(csbm, embeddings, densities)
