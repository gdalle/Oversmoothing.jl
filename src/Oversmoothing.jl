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

using CairoMakie
using Colors
using DensityInterface: DensityInterface, logdensityof, densityof
using IntervalArithmetic: interval, inf, sup
using LogExpFunctions: logsumexp
using MonteCarloMeasurements: Particles
using OffsetArrays: OffsetArray, OffsetMatrix, OffsetVector, Origin
using StatsBase: StatsBase, entropy, kldivergence, sample
using StatsFuns: binompdf, log2π, normpdf, normlogpdf

const kl = StatsBase.kldivergence

## includes

include("categorical.jl")
include("normal.jl")
include("mixture.jl")

include("kl.jl")
include("entropy.jl")
include("error.jl")

include("sbm.jl")
include("embeddings.jl")
include("first_layer.jl")

include("plot.jl")

## exports

export MultivariateNormal
export Mixture
export StochasticBlockModel, SBM
export ContextualStochasticBlockModel, CSBM
export community_size, community_range, community_of_vertex, nb_vertices, nb_communities
export embeddings
export first_layer_mixtures
export kl_interval, kl_montecarlo
export entropy_interval, entropy_montecarlo
export error_interval, error_montecarlo
export plot_1d

end # module Oversmoothing
