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
using HCubature: HCubature
using LogExpFunctions: logsumexp, softmax
using MLJLinearModels: MLJLinearModels, MultinomialRegression
using ProgressLogging: @progress
using QuadGK: QuadGK
using StaticArrays: SVector, SMatrix, @SVector
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π

## includes

include("uncertainty.jl")

include("normal.jl")
include("mixture.jl")
include("accuracy.jl")

include("sbm.jl")
include("embeddings.jl")
include("first_layer.jl")
include("random_walk.jl")
include("logistic_regression.jl")
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
export LinearCSBM1d, CircularCSBM2d
export empirical_embeddings
export accuracy_quadrature, accuracy_montecarlo
export first_layer_densities, random_walk_densities
export accuracy_zeroth_layer, accuracy_first_layer, random_walk_accuracy_trajectories
export logistic_regression_accuracy_trajectories
export accuracy_by_depth
export plot_1d, plot_2d

end # module Oversmoothing
