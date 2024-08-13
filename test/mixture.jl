using DensityInterface
using LinearAlgebra
using MonteCarloMeasurements
using Oversmoothing
using StableRNGs
using Test

rng = StableRNG(63)

## Univariate

mix = Mixture([UnivariateNormal(1.0, 1.0), UnivariateNormal(4.0, 2.0)], [0.4, 0.6])

samples = [rand(rng, mix) for _ in 1:10000];
@test mean(mix) ≈ mean(samples) rtol = 1e-1
@test cov(mix) ≈ cov(samples) rtol = 1e-1
@test densityof(mix, zeros(1)) ≈
    dot(mix.weights, densityof.(mix.distributions, Ref(zeros(1))))

@test pmean(error_montecarlo(rng, mix; nb_samples=1000)) ≈ error_quadrature(mix) rtol = 1e-1

## Bivariate

mix = Mixture(
    [
        BivariateNormal([1.0, 2.0], [1.0 0.2; 0.2 2.0]),
        BivariateNormal([4.0, 3.0], [2.0 0.1; 0.1 1.0]),
    ],
    [0.4, 0.6],
)

samples = [rand(rng, mix) for _ in 1:10000];
@test mean(mix) ≈ mean(samples) rtol = 1e-1
@test cov(mix) ≈ cov(samples) rtol = 1e-1
@test densityof(mix, zeros(2)) ≈
    dot(mix.weights, densityof.(mix.distributions, Ref(zeros(2))))

@test pmean(error_montecarlo(rng, mix; nb_samples=1000)) ≈ error_quadrature(mix) rtol = 1e-1
