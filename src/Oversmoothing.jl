module Oversmoothing

using Base.Threads: @threads, nthreads
using BlockArrays: Block, BlockArray, undef_blocks
using DensityInterface: DensityInterface, IsDensity, densityof, logdensityof
using Graphs: adjacency_matrix, erdos_renyi, stochastic_block_model
using LinearAlgebra:
    BLAS, Cholesky, Diagonal, I, Symmetric, cholesky, det, dot, inv, logdet, mul!
using LogExpFunctions: logsumexp
using OhMyThreads: tmap, tforeach
using Random: Random, AbstractRNG, default_rng, rand!, randn!, randsubseq
using SparseArrays: SparseMatrixCSC, nonzeros, nnz, sparse, sprand
using Statistics: Statistics, mean, var
using StatsBase: StatsBase, sample
using StatsFuns: binompdf, log2π, normpdf, normlogpdf

abstract type AbstractRandomGraph end
abstract type AbstractMeasure end
abstract type AbstractConvolution end

@inline DensityInterface.DensityKind(::AbstractMeasure) = IsDensity()

include("univariategaussian.jl")
include("multivariategaussian.jl")
include("mixture.jl")
include("erdosrenyi.jl")
include("stochasticblockmodel.jl")
include("contextual.jl")
include("convolution.jl")
include("embeddings.jl")

Gaussian(μ::Number, σ²::Number) = UnivariateGaussian(μ, σ²)
Gaussian(μ::AbstractVector, Σ::AbstractMatrix) = MultivariateGaussian(μ, Σ)

export AbstractMeasure, AbstractRandomGraph, AbstractConvolution
export Gaussian, UnivariateGaussian, MultivariateGaussian, Mixture
export bernoulli_matrix
export ErdosRenyi, ER, StochasticBlockModel, SBM
export Contextual
export NeighborhoodAverage, NeighborhoodSum
export nb_communities, get_community, community_size
export embedding_samples, embedding_samples_indep, split_by_community

end # module Oversmoothing
