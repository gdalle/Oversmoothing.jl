function quadrature_1d(f; rtol=1e-5, bound=10, kwargs...)
    result, error = QuadGK.quadgk(f, -bound, bound; rtol, kwargs...)
    return result
end

function quadrature_nd(f, dim; rtol=1e-5, bound=10, kwargs...)
    result, error = HCubature.hcubature(
        f, -fill(bound, dim), +fill(bound, dim); rtol, kwargs...
    )
    return result
end

struct BayesAccuracy{M<:Mixture}
    mix::M
    normalized::Bool
end

function (ba::BayesAccuracy)(x)
    d, w = distributions(ba.mix), weights(ba.mix)
    p, t = 0.0, 0.0
    for i in eachindex(w, d)
        l = w[i] * densityof(d[i], x)
        p = max(p, l)
        t += l
    end
    if ba.normalized
        return p / t
    else
        return p
    end
end

function accuracy_montecarlo(rng::AbstractRNG, mix::Mixture; nb_samples)
    ba = BayesAccuracy(mix, true)
    x = rand(rng, mix, nb_samples)
    accuracy_samples = fill(NaN, nb_samples)
    for s in 1:nb_samples
        accuracy_samples[s] = ba(x[s])
    end
    return MonteCarloValue(accuracy_samples)
end

function accuracy_quadrature(mix::Mixture; kwargs...)
    f = BayesAccuracy(mix, false)
    if length(mix) == 1
        return quadrature_1d(f; kwargs...)
    else
        return quadrature_nd(f, length(mix); kwargs...)
    end
end

struct AbsoluteDifference{D1,D2}
    f::D1
    g::D2
end

(ad::AbsoluteDifference)(x) = abs(densityof(ad.f, x) - densityof(ad.g, x)) / 2

function total_variation_quadrature(density1, density2; kwargs...)
    f = AbsoluteDifference(density1, density2)
    if length(density1) == 1
        return quadrature_1d(f; kwargs...)
    else
        return quadrature_nd(f, length(density1); kwargs...)
    end
end
