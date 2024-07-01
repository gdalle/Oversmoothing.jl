function bayes_classification_error(
    rng::AbstractRNG, dists::AbstractVector, p::AbstractVector{<:Real}; nb_samples=100
)
    error_bools = map(1:nb_samples) do i
        c_true = sample(rng, eachindex(dists), StatsBase.weights(p))
        x = rand(rng, dists[c_true])
        l = logdensityof(dists[c_true], x)
        for c in eachindex(dists)
            if c != c_true && logdensityof(dists[c], x) > l
                return true
            else
                return false
            end
        end
    end
    return mean(error_bools)
end

function bayes_classification_error_interval(
    rng::AbstractRNG,
    dists::AbstractVector,
    p::AbstractVector{<:Real};
    nb_samples=100,
    nb_errors=100,
)
    errors = map(1:nb_errors) do k
        bayes_classification_error(rng, dists, p; nb_samples)
    end
    return quantile(errors, 0.05), quantile(errors, 0.95)
end
