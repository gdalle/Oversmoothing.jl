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
    accuracy_samples = fill(NaN, nb_samples)
    for s in 1:nb_samples
        x = rand(rng, mix)
        accuracy_samples[s] = ba(x)
    end
    return MonteCarloValue(accuracy_samples)
end

function accuracy_quadrature_1d(mix::Mixture; rtol, bound=100, kwargs...)
    ba = BayesAccuracy(mix, false)
    accuracy, quad_accuracy = QuadGK.quadgk(ba, -bound, bound; rtol, kwargs...)
    return IntervalValue(accuracy, quad_accuracy)
end

function accuracy_quadrature_nd(mix::Mixture; rtol, bound=100, kwargs...)
    ba = BayesAccuracy(mix, false)
    accuracy, cub_accuracy = HCubature.hcubature(
        ba, -fill(bound, length(mix)), +fill(bound, length(mix)); rtol, kwargs...
    )
    return IntervalValue(accuracy, cub_accuracy)
end

function accuracy_quadrature(mix::Mixture; rtol, kwargs...)
    if length(mix) == 1
        return accuracy_quadrature_1d(mix; rtol, kwargs...)
    else
        return accuracy_quadrature_nd(mix; rtol, kwargs...)
    end
end
