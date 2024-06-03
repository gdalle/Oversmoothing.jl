function kl(dist_f::MvNormal, dist_g::MvNormal)
    d = length(dist_f)
    μf, μg = mean(dist_f), mean(dist_g)
    Σf, Σg = _cov(dist_f).mat, _cov(dist_g).mat
    term1 = logdet(Σg) - logdet(Σf)
    term2 = tr(Σg \ Σf)
    term3 = dot(μf - μg, inv(Σg), μf - μg)
    return (term1 + term2 + term3 - d) / 2
end

function log_prod_norm(dist_a::MvNormal, dist_b::MvNormal)
    d = length(dist_a)
    μa, μb = mean(dist_a), mean(dist_b)
    Σa, Σb = _cov(dist_a), _cov(dist_b)
    term1 = -d * log(2π)
    term2 = -logdet(Σa + Σb)
    term3 = -dot(μb - μa, inv(Σa + Σb), μb - μa)
    return (term1 + term2 + term3) / 2
end

prod_norm(ga::MvNormal, gb::MvNormal) = exp(log_prod_norm(ga, gb))

function kl_lowerbound(f::MvNormalMixture, g::MvNormalMixture)
    ωf, ωg = probs(f), probs(g)
    cf, cg = components(f), components(g)
    L = 0.0
    for a in eachindex(ωf, cf)
        num = sum(ωf[α] * exp(-kl(cf[a], cf[α])) for α in eachindex(ωf, cf))
        den = sum(ωg[b] * prod_norm(cf[a], cg[b]) for b in eachindex(ωg, cg))
        L += ωf[a] * (log(num / den) - entropy(cf[a]))
    end
    return L
end

function kl_upperbound(f::MvNormalMixture, g::MvNormalMixture)
    ωf, ωg = probs(f), probs(g)
    cf, cg = components(f), components(g)
    U = 0.0
    for a in eachindex(ωf, cf)
        num = sum(ωf[α] * prod_norm(cf[a], cf[α]) for α in eachindex(ωf, cf))
        den = sum(ωg[b] * exp(-kl(cf[a], cg[b])) for b in eachindex(ωg, cg))
        U += ωf[a] * (log(num / den) + entropy(cf[a]))
    end
    return U
end

function kl_approx(f::MvNormalMixture, g::MvNormalMixture)
    return (kl_lowerbound(f, g) + kl_upperbound(f, g)) / 2
end

function kl_empirical(
    rng::AbstractRNG, f::MvNormalMixture, g::MvNormalMixture; nb_samples=1000
)
    x = rand(rng, f, nb_samples)
    log_fx = logpdf(f, x)
    log_gx = logpdf(g, x)
    kl_mean = mean(log_fx .- log_gx)
    kl_std = std(log_fx .- log_gx)
    return kl_mean, kl_std
end
