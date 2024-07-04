using DensityInterface
using LinearAlgebra
using Oversmoothing
using StableRNGs
using Test

rng = StableRNG(63)

mix = Mixture(
    [
        MultivariateNormal([1.0, 2.0], [1.0 0.2; 0.2 0.5]),
        MultivariateNormal([4.0, 3.0], [1.0 0.2; 0.2 0.5]),
    ],
    [0.4, 0.6],
)

samples = [rand(rng, mix) for _ in 1:10000];
@test mean(mix) ≈ mean(samples) rtol = 1e-1
@test cov(mix) ≈ cov(samples) rtol = 1e-1
@test densityof(mix, zeros(2)) ≈
    dot(mix.weights, densityof.(mix.distributions, Ref(zeros(2))))
