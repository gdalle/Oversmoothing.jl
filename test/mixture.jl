using DensityInterface
using LinearAlgebra
using Oversmoothing
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

## Univariate

mix = Mixture([UnivariateNormal(1.0, 1.0), UnivariateNormal(4.0, 2.0)], [0.4, 0.6])

x = rand(rng, mix)
@test x isa AbstractVector
@test densityof(mix, x) ≈ exp(logdensityof(mix, x))

x = rand(rng, mix, 10000)
@test mean(x) ≈ mean(mix) rtol = 1e-1
@test cov(x) ≈ cov(mix) rtol = 1e-1

@test value(accuracy_montecarlo(rng, mix; nb_samples=1000)) ≈
    accuracy_quadrature(mix; rtol=1e-2) rtol = 1e-1

@test total_variation_quadrature(mix, mix; rtol=1e-2) == 0
@test 0 < total_variation_quadrature(mix, MultivariateNormal(mix); rtol=1e-2) < 1

## Bivariate

mix = Mixture(
    [
        BivariateNormal([1.0, 2.0], [1.0 0.2; 0.2 2.0]),
        BivariateNormal([4.0, 3.0], [2.0 0.1; 0.1 1.0]),
    ],
    [0.4, 0.6],
)

x = rand(rng, mix)
@test x isa AbstractVector
@test densityof(mix, x) ≈ exp(logdensityof(mix, x))

x = rand(rng, mix, 10000)
@test mean(x) ≈ mean(mix) rtol = 1e-1
@test cov(x) ≈ cov(mix) rtol = 1e-1

@test value(accuracy_montecarlo(rng, mix; nb_samples=1000)) ≈
    accuracy_quadrature(mix; rtol=1e-2) rtol = 1e-1

@test total_variation_quadrature(mix, mix; rtol=1e-2) == 0
@test 0 < total_variation_quadrature(mix, MultivariateNormal(mix); rtol=1e-2) < 1
