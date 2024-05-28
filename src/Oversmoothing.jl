module Oversmoothing

using Base.Threads: @threads, nthreads
using CairoMakie
using DensityInterface: DensityInterface, IsDensity, densityof, logdensityof
using Distributions: MultivariateDistribution, pdf
using KernelDensity: kde
using LinearAlgebra:
    BLAS,
    Cholesky,
    Diagonal,
    I,
    Symmetric,
    checksquare,
    det,
    dot,
    inv,
    issymmetric,
    ldiv!,
    logdet,
    mul!
using LogarithmicNumbers: LogFloat64, Logarithmic
using LogExpFunctions: logsumexp
using OhMyThreads: tmap, tforeach
using ProgressMeter: Progress, next!
using Random: Random, AbstractRNG, default_rng, rand!, randn!, randsubseq
using SparseArrays: SparseMatrixCSC, nonzeros, nnz, sparse, sprand, spzeros
using StableRNGs: StableRNG
using Statistics: Statistics, mean, std, var
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π, normpdf, normlogpdf

include("mixture.jl")
include("random_graph.jl")
include("erdos_renyi.jl")
include("stochastic_block_model.jl")
include("embeddings.jl")
include("distances.jl")
include("depth.jl")
include("plot.jl")

export AbstractRandomGraph
export Mixture
export ErdosRenyi, ER, StochasticBlockModel, SBM
export community_size, community_range, community_of_vertex
export embeddings, split_by_community
export density_estimator, empirical_kl, misclassification_probability
export plot_1d_embeddings, plot_2d_embeddings
export plot_misclassification

end # module Oversmoothing
