module Oversmoothing

using Base.Threads: @threads, nthreads
using BlockArrays: Block, BlockArray, undef_blocks
using DensityInterface: DensityInterface, IsDensity, densityof, logdensityof
using Distributions: MultivariateDistribution, pdf
using KernelDensity: kde
using LinearAlgebra:
    BLAS, Cholesky, Diagonal, I, Symmetric, cholesky, det, dot, inv, ldiv!, logdet, mul!
using LogExpFunctions: logsumexp
using OhMyThreads: tmap, tforeach
using Random: Random, AbstractRNG, default_rng, rand!, randn!, randsubseq
using SparseArrays: SparseMatrixCSC, nonzeros, nnz, sparse, sprand
using Statistics: Statistics, mean, var
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π, normpdf, normlogpdf

include("mixture.jl")
include("random_graph.jl")
include("erdos_renyi.jl")
include("stochastic_block_model.jl")
include("embeddings.jl")
include("distances.jl")

export AbstractRandomGraph
export Mixture
export bernoulli_matrix
export ErdosRenyi, ER, StochasticBlockModel, SBM
export embeddings, split_by_community
export empirical_kl

end # module Oversmoothing
