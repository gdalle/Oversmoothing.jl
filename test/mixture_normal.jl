using LinearAlgebra
using Oversmoothing
using StableRNGs
using Test

rng = StableRNG(63)

function create_mixture(rng; d, k)
    cs = [MultivariateNormal(randn(rng, d), Diagonal(rand(rng, d))) for _ in 1:k]
    ws = rand(rng, k)
    ws ./= sum(ws)
    return Mixture(cs, ws)
end

mix1 = create_mixture(rng; d=3, k=2)
mix2 = create_mixture(rng; d=3, k=2)

@test kl_lowerbound(mix1, mix2) <= kl_approx(mix1, mix2)
@test kl_upperbound(mix1, mix2) >= kl_approx(mix1, mix2)

@test kl_lowerbound(mix1, mix2) <= kl_empirical(rng, mix1, mix2)
@test kl_upperbound(mix1, mix2) >= kl_empirical(rng, mix1, mix2)
