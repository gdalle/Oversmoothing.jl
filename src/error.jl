function error_montecarlo(
    rng::AbstractRNG, mix::Mixture; nb_dist_samples=100, nb_error_samples=100
)
    d, w = distributions(mix), weights(mix)
    error_fractions = map(1:nb_error_samples) do _
        error_bools = map(1:nb_dist_samples) do i
            i_true = sample(rng, eachindex(d), StatsBase.weights(w))
            x = rand(rng, d[i_true])
            l = logdensityof(d[i_true], x)
            for i in eachindex(d)
                if i != i_true && logdensityof(d[i], x) > l
                    return true
                else
                    return false
                end
            end
        end
        mean(error_bools)
    end
    return Particles(error_fractions)
end

function jensen_shannon_interval(mix::Mixture)
    return entropy_interval(flat(mix)) -
           dot(weights(mix), entropy_interval.(distributions(mix)))
end

"""
    error_interval(mix)

# Reference

> Divergence measures based on the Shannon entropy
"""
function error_interval(mix::Mixture)
    n = length(mix)
    π = weights(mix)
    diff = entropy(Categorical(π)) - jensen_shannon_interval(mix)
    U = sup(diff) / 2
    L = inf(diff)^2 / (4(n - 1))
    return interval(L, U)
end
