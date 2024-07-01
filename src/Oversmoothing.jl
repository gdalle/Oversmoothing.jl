module Oversmoothing

## stdlibs

using Base.Threads: @threads, nthreads
using LinearAlgebra
using LinearAlgebra: checksquare
using Statistics
using Random
using Random: AbstractRNG, default_rng, rand!, randn!, randsubseq
using SparseArrays

## other deps

using DensityInterface: DensityInterface, logdensityof, densityof
using LogExpFunctions: logsumexp
using OffsetArrays: OffsetArray, OffsetMatrix, OffsetVector, Origin
using StatsBase: StatsBase, entropy, kldivergence, sample
using StatsFuns: binompdf, log2π, normpdf, normlogpdf

## includes

include("normal.jl")
include("mixture.jl")
include("kl.jl")
include("error.jl")
include("stochastic_block_model.jl")
include("embeddings.jl")
include("state_evolution.jl")

## exports

export MultivariateNormal
export Mixture
export StochasticBlockModel, SBM
export community_size, community_range, community_of_vertex, nb_vertices, nb_communities
export embeddings, split_by_community
export first_layer_mixtures
export kl_lowerbound, kl_upperbound, kl_approx, kl_empirical
export bayes_classification_error, bayes_classification_error_interval

end # module Oversmoothing
