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

using Distributions:
    Distributions,
    Binomial,
    Categorical,
    Continuous,
    MixtureModel,
    Multivariate,
    MultivariateDistribution,
    MvNormal,
    component,
    components,
    _cov,
    entropy,
    logpdf,
    pdf,
    probs
using LogarithmicNumbers: LogFloat64, Logarithmic
using LogExpFunctions: logsumexp
using OffsetArrays: OffsetArray, OffsetMatrix, OffsetVector, Origin
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π, normpdf, normlogpdf

## includes

include("stochastic_block_model.jl")
include("embeddings.jl")
include("mixture.jl")
include("state_evolution.jl")
include("kullback_leibler.jl")

## exports

export StochasticBlockModel, SBM
export community_size, community_range, community_of_vertex, nb_vertices, nb_communities
export embeddings, split_by_community
export state_evolution
export kl_lowerbound, kl_upperbound, kl_approx, kl_empirical

end # module Oversmoothing
