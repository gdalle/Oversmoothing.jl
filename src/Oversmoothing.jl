module Oversmoothing

using Base.Threads: @threads, nthreads
using BlockArrays: Block, BlockArray, undef_blocks
using CairoMakie:
    CairoMakie,
    Aspect,
    Axis,
    Colorbar,
    Figure,
    Label,
    Legend,
    axislegend,
    colsize!,
    contour,
    contour!,
    hexbin,
    hexbin!,
    hidedecorations!,
    hist,
    hist!,
    lines,
    lines!,
    resize_to_layout!,
    linkaxes!,
    linkxaxes!,
    linkyaxes!
using DensityInterface: DensityInterface, IsDensity, densityof, logdensityof
using Distributions: MultivariateDistribution, pdf
using KernelDensity: kde
using LinearAlgebra:
    BLAS, Cholesky, Diagonal, I, Symmetric, cholesky, det, dot, inv, ldiv!, logdet, mul!
using LogarithmicNumbers: LogFloat64, Logarithmic
using LogExpFunctions: logsumexp
using OhMyThreads: tmap, tforeach
using Random: Random, AbstractRNG, default_rng, rand!, randn!, randsubseq
using SparseArrays: SparseMatrixCSC, nonzeros, nnz, sparse, sprand
using StableRNGs: StableRNG
using Statistics: Statistics, mean, var
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π, normpdf, normlogpdf

include("mixture.jl")
include("random_graph.jl")
include("erdos_renyi.jl")
include("stochastic_block_model.jl")
include("embeddings.jl")
include("distances.jl")
include("plot.jl")

export AbstractRandomGraph
export Mixture
export bernoulli_matrix
export ErdosRenyi, ER, StochasticBlockModel, SBM
export community_size, community_range, community_of_vertex
export embeddings, split_by_community
export density_estimator, empirical_kl
export plot_1d_embeddings, plot_2d_embeddings

end # module Oversmoothing
