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
    C = length(mix)
    π = weights(mix)
    conditional_entropy = entropy(Categorical(π)) - jensen_shannon_interval(mix)
    # Jensen-Shannon upper bound
    U = sup(conditional_entropy) / 2
    # Fano lower bound
    L = fano(log(C - 1), inf(conditional_entropy))
    return interval(L, U)
end

H(p::Real) = -xlogx(p) - xlogx(1 - p)

fano_aux(p::Real, (a, b)) = H(p) + a * p - b

function fano(a, b)
    zs = find_zeros(Base.Fix2(fano_aux, (a, b)), 0.0, 1.0)
    if isempty(zs)
        if fano_aux(0, (a, b)) > 0
            return 0.0
        else
            error()
        end
    else
        return minimum(zs)
    end
end
