const MultivariateNormalMixture = Mixture{<:MultivariateNormal}

function kl_lowerbound(f::MultivariateNormalMixture, g::MultivariateNormalMixture)
    ωf, ωg = weights(f), weights(g)
    cf, cg = components(f), components(g)
    L = 0.0
    for a in eachindex(ωf, cf)
        num = sum(ωf[α] * exp(-kldivergence(cf[a], cf[α])) for α in eachindex(ωf, cf))
        den = sum(ωg[b] * prod_norm(cf[a], cg[b]) for b in eachindex(ωg, cg))
        L += ωf[a] * (log(num / den) - entropy(cf[a]))
    end
    return L
end

function kl_upperbound(f::MultivariateNormalMixture, g::MultivariateNormalMixture)
    ωf, ωg = weights(f), weights(g)
    cf, cg = components(f), components(g)
    U = 0.0
    for a in eachindex(ωf, cf)
        num = sum(ωf[α] * prod_norm(cf[a], cf[α]) for α in eachindex(ωf, cf))
        den = sum(ωg[b] * exp(-kldivergence(cf[a], cg[b])) for b in eachindex(ωg, cg))
        U += ωf[a] * (log(num / den) + entropy(cf[a]))
    end
    return U
end

function kl_approx(f::MultivariateNormalMixture, g::MultivariateNormalMixture)
    return (kl_lowerbound(f, g) + kl_upperbound(f, g)) / 2
end

function kl_empirical(
    rng::AbstractRNG,
    f::MultivariateNormalMixture,
    g::MultivariateNormalMixture;
    nb_samples=1000,
)
    x = rand(rng, f, nb_samples)
    log_fx = logdensityof.(Ref(f), x)
    log_gx = logdensityof.(Ref(g), x)
    return mean(log_fx .- log_gx)
end
