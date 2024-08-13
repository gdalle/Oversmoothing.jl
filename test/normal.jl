using DensityInterface
using Distributions: Distributions
using LinearAlgebra
using Oversmoothing
using StableRNGs
using StatsBase
using Test

rng = StableRNG(63)

μ = randn(rng, 2)
L = randn(rng, 2, 2)
Σ = L * transpose(L)
@test isposdef(Σ)

g = BivariateNormal(μ, Σ)
g_ref = Distributions.MvNormal(μ, Σ)

x = rand(rng, g)
@test x isa AbstractVector
@test logdensityof(g, x) ≈ Distributions.logpdf(g_ref, x)
