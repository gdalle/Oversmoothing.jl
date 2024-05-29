using Distributions
using DensityInterface
using LinearAlgebra
using Oversmoothing
using StableRNGs
using Test

rng = StableRNG(63)

components = [MvNormal([0.0], [0.5;;]), MvNormal([1.0], [1.0;;]), MvNormal([2.0], [3.0;;])]
weights = [0.2, 0.5, 0.3]
mix = Mixture(components, weights)

samples = [rand(rng, mix) for _ in 1:10000];
@test mean(mix) ≈ mean(samples) rtol = 1e-1
@test cov(mix) ≈ cov(samples) rtol = 1e-1
@test densityof(mix, [0]) ≈ dot(weights, densityof.(components, Ref([0])))
