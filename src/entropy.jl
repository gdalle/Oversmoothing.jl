function entropy_montecarlo(rng::AbstractRNG, f::MultivariateNormalMixture; nb_samples=100)
    x = rand(rng, f, nb_samples)
    log_fx = logdensityof.(Ref(f), x)
    return Particles(-log_fx)
end

"""
    entropy_interval(mix)

# References

> Lower and upper bounds for approximation of the Kullback-Leibler divergence between Gaussian Mixture Models
"""
function entropy_interval(f::MultivariateNormalMixture)
    ωf = weights(f)
    df = distributions(f)
    inds = eachindex(ωf, df)
    L, U = 0.0, 0.0

    for a in inds
        L += -ωf[a] * log(sum(ωf[α] * prod_norm(df[a], df[α]) for α in inds))
        U +=
            -ωf[a] * log(sum(ωf[α] * exp(-kl(df[a], df[α])) for α in inds)) +
            ωf[a] * entropy(df[a])
    end
    return interval(L, U)
end

entropy_interval(f::MultivariateNormal) = entropy(f)
