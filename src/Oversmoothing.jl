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
using Flux: Flux, Adam, DataLoader, Dense, onehotbatch, relu, softmax
using Flux.Losses: crossentropy
using GraphNeuralNetworks: GraphNeuralNetworks, GNNChain, GNNGraph, GNNLayer, SGConv
using HCubature: HCubature
using LogExpFunctions: logsumexp
using ProgressLogging: @progress
using QuadGK: QuadGK
using StaticArrays: SVector, SMatrix, @SVector
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π
using Zygote: gradient

## includes

include("uncertainty.jl")
include("normal.jl")
include("mixture.jl")
include("accuracy.jl")

include("sbm.jl")
include("embeddings.jl")
include("first_layer.jl")
include("random_walk.jl")
include("gnn.jl")
include("depth.jl")

function plot_1d end
function plot_2d end

## exports

export value, uncertainty
export MultivariateNormal, UnivariateNormal, BivariateNormal
export Mixture
export StochasticBlockModel, SBM
export ContextualStochasticBlockModel, CSBM
export nb_vertices, nb_communities
export community_size, community_range, community_of_vertex
export embeddings
export accuracy_quadrature, accuracy_montecarlo
export first_layer_densities
export random_walk_densities, random_walk_accuracy_trajectories
export gnn_accuracy_trajectories
export accuracy_zeroth_layer, accuracy_first_layer
export accuracy_by_depth
export plot_1d, plot_2d

end # module Oversmoothing
