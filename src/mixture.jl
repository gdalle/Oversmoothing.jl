struct Mixture{D,W}
    distributions::Vector{D}
    weights::Vector{W}

    function Mixture(d::Vector{D}, w::Vector{W}) where {D,W}
        @assert sum(w) ≈ one(W)
        @assert length(d) == length(w)
        return new{D,W}(d, w)
    end
end

const MultivariateNormalMixture = Mixture{<:MultivariateNormal}

@inline DensityInterface.DensityKind(::Mixture) = DensityInterface.HasDensity()

Base.length(mix::Mixture) = length(mix.distributions)
distributions(mix::Mixture) = mix.distributions
weights(mix::Mixture) = mix.weights

function Base.rand(rng::AbstractRNG, mix::Mixture)
    di = sample(rng, distributions(mix), StatsBase.weights(weights(mix)))
    return rand(rng, di)
end

function Base.rand(rng::AbstractRNG, mix::Mixture, n::Integer)
    dis = sample(rng, distributions(mix), StatsBase.weights(weights(mix)), n)
    return [rand(rng, di) for di in dis]
end

function Statistics.mean(mix::Mixture)
    d, w = distributions(mix), weights(mix)
    return sum(w[i] * mean(d[i]) for i in eachindex(d, w))
end

function squared_mean(m)
    μ = mean(m)
    return μ * transpose(μ)
end

function Statistics.cov(mix::Mixture)
    d, w = distributions(mix), weights(mix)
    second_moment = sum(w[i] * (cov(d[i]) + squared_mean(d[i])) for i in eachindex(d, w))
    Σ = second_moment - squared_mean(mix)
    return Σ
end

function DensityInterface.densityof(mix::Mixture, x)
    d, w = distributions(mix), weights(mix)
    return sum(w[i] * densityof(d[i], x) for i in eachindex(d, w))
end

function DensityInterface.logdensityof(mix::Mixture, x)
    d, w = distributions(mix), weights(mix)
    return logsumexp(log(w[i]) + logdensityof(d[i], x) for i in eachindex(d, w))
end

function compress(mix::Mixture)
    d, w = distributions(mix), weights(mix)
    to_keep = trues(length(mix))
    for j in eachindex(d, w)
        for i in eachindex(d, w)
            i < j || continue
            if isapprox(d[i], d[j]) && to_keep[i]
                w[i] += w[j]
                to_keep[j] = false
                break
            end
        end
    end
    return Mixture(d[to_keep], w[to_keep])
end
