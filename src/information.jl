## Entropy

function entropy_montecarlo(
    rng::AbstractRNG, mix::MultivariateNormalMixture; nb_samples=100
)
    x = rand(rng, mix, nb_samples)
    log_fx = logdensityof.(Ref(mix), x)
    return Particles(-log_fx)
end

"""
    entropy_interval(mix)

# References

> Estimating Mixture Entropy with Pairwise Distances
"""
function entropy_interval(mix::MultivariateNormalMixture)
    c = weights(mix)
    p = distributions(mix)
    inds = eachindex(c, p)
    U = L = sum(c[i] * entropy(p[i]) for i in inds)
    for i in inds
        L -= c[i] * log(sum(c[j] * exp(-chernoff(p[i], p[j])) for j in inds))
        U -= c[i] * log(sum(c[j] * exp(-kl(p[i], p[j])) for j in inds))
    end
    return interval(L, U)
end

entropy_interval(f::MultivariateNormal) = entropy(f)
