using DensityInterface
using Distributions: Distributions
using LinearAlgebra
using Oversmoothing
using StableRNGs
using StatsBase
using Test

rng = StableRNG(63)

k = 3
μ = randn(rng, k)
L = randn(rng, k, k)
Σ = L * transpose(L)
@test isposdef(Σ)

g = MultivariateNormal(μ, Σ)
g_ref = Distributions.MvNormal(μ, Σ)

g2 = MultivariateNormal(2μ, Σ + I)
g2_ref = Distributions.MvNormal(2μ, Σ + I)

x = rand(rng, g)
@test x isa AbstractVector
@test logdensityof(g, x) ≈ Distributions.logpdf(g_ref, x)

X = rand(rng, g, 10)
@test X isa AbstractMatrix
