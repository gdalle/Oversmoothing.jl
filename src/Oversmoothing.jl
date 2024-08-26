module Oversmoothing

## stdlibs

using Base.Threads: @threads, nthreads
using LinearAlgebra
using LinearAlgebra: LowerTriangular, checksquare, ishermitian
using Statistics
using Random
using Random: AbstractRNG, default_rng, rand!, randn!, randsubseq
using SparseArrays

## other deps

using DensityInterface: DensityInterface, logdensityof, densityof
using Flux: Flux, Dense, onehotbatch, relu, softmax
using Flux.Losses: crossentropy
using GraphNeuralNetworks: GNNChain, GNNGraph, SGConv
using HCubature: HCubature
using LogExpFunctions: logsumexp
using MonteCarloMeasurements: Particles, pmean
using OhMyThreads: OhMyThreads, tforeach
using QuadGK: QuadGK
using StaticArrays: SVector, SMatrix, @SVector
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π
using Zygote: gradient

## includes

include("normal.jl")
include("mixture.jl")
include("error.jl")

include("sbm.jl")
include("embeddings.jl")
include("first_layer.jl")
include("random_walk.jl")
include("gnn.jl")
include("depth.jl")

function plot_1d end
function plot_2d end

## exports

export MultivariateNormal, UnivariateNormal, BivariateNormal
export Mixture
export StochasticBlockModel, SBM
export ContextualStochasticBlockModel, CSBM
export nb_vertices, nb_communities
export community_size, community_range, community_of_vertex
export embeddings
export error_quadrature, error_montecarlo
export first_layer_densities
export random_walk_densities, random_walk_error_trajectories
export gcn_error_trajectories
export error_zeroth_layer, error_first_layer
export error_by_depth, optimal_depth
export plot_1d, plot_2d

end # module Oversmoothing
