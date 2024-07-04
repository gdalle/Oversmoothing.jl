function kl_montecarlo(
    rng::AbstractRNG,
    f::MultivariateNormalMixture,
    g::MultivariateNormalMixture;
    nb_samples=100,
)
    x = rand(rng, f, nb_samples)
    log_fx = logdensityof.(Ref(f), x)
    log_gx = logdensityof.(Ref(g), x)
    return Particles(log_fx .- log_gx)
end

"""
    kl_interval(mix1, mix2)

# Reference

> Lower and upper bounds for approximation of the Kullback-Leibler divergence between Gaussian Mixture Models
"""
function kl_interval(f::MultivariateNormalMixture, g::MultivariateNormalMixture)
    ωf, ωg = weights(f), weights(g)
    df, dg = distributions(f), distributions(g)
    L, U = 0.0, 0.0

    for a in eachindex(ωf, df)
        L_num = sum(ωf[α] * exp(-kl(df[a], df[α])) for α in eachindex(ωf, df))
        L_den = sum(ωg[b] * prod_norm(df[a], dg[b]) for b in eachindex(ωg, dg))

        U_num = sum(ωf[α] * prod_norm(df[a], df[α]) for α in eachindex(ωf, df))
        U_den = sum(ωg[b] * exp(-kl(df[a], dg[b])) for b in eachindex(ωg, dg))

        L += ωf[a] * (log(L_num / L_den) - entropy(df[a]))
        U += ωf[a] * (log(U_num / U_den) + entropy(df[a]))
    end
    return interval(L, U)
end

kl_interval(f::MultivariateNormal, g::MultivariateNormal) = kl(f, g)
