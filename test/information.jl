using IntervalArithmetic: in_interval
using LinearAlgebra
using MonteCarloMeasurements: pmean
using Oversmoothing
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

function create_mixture(rng; dim, nb_components)
    ds = [
        MultivariateNormal(randn(rng, dim), Diagonal(rand(rng, dim))) for
        _ in 1:nb_components
    ]
    ws = rand(rng, nb_components)
    ws ./= sum(ws)
    return Mixture(ds, ws)
end

mix1 = create_mixture(rng; dim=3, nb_components=2)
mix2 = create_mixture(rng; dim=3, nb_components=2)

ent_mc = pmean(entropy_montecarlo(rng, mix1; nb_samples=1000))
ent_int = entropy_interval(mix1)
@test in_interval(ent_mc, ent_int)

err_mc = pmean(error_montecarlo(rng, mix1))
err_int = error_interval(mix1)
@test in_interval(err_mc, err_int)
