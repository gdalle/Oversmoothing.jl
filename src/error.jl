function error_montecarlo(
    rng::AbstractRNG, mix::Mixture; nb_dist_samples=100, nb_error_samples=100
)
    d, w = distributions(mix), weights(mix)
    error_fractions = map(1:nb_error_samples) do _
        error_bools = map(1:nb_dist_samples) do i
            i_true = sample(rng, eachindex(d), StatsBase.weights(w))
            x = rand(rng, d[i_true])
            p_true = w[i_true] * logdensityof(d[i_true], x)
            for i in eachindex(d)
                if i != i_true && w[i] * logdensityof(d[i], x) > p_true
                    return true
                end
            end
            return false
        end
        mean(error_bools)
    end
    return Particles(error_fractions)
end

struct BayesPrecision{M<:Mixture}
    mix::M
end

function (bp::BayesPrecision)(x)
    d, w = distributions(bp.mix), weights(bp.mix)
    p = w .* densityof.(d, Ref(x))
    return maximum(p)
end

function error_quadrature_1d(mix::Mixture; kwargs...)
    bp = BayesPrecision(mix)
    quad_result, quad_error = QuadGK.quadgk(bp, -Inf, Inf; kwargs...)
    precision = interval(quad_result - quad_error, quad_result + quad_error)
    return 1 - precision
end

function error_quadrature_2d(mix::Mixture; bound=100, kwargs...)
    bp = BayesPrecision(mix)
    cub_result, cub_error = HCubature.hcubature(
        bp, (-bound, -bound), (bound, bound); kwargs...
    )
    precision = interval(cub_result - cub_error, cub_result + cub_error)
    return 1 - precision
end
