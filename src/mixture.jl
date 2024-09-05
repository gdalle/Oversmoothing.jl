struct Mixture{D,W}
    distributions::Vector{D}
    weights::Vector{W}

    function Mixture(d::Vector{D}, w::Vector{W}) where {D,W}
        @assert sum(w) ≈ one(W)
        @assert length(d) == length(w)
        return new{D,W}(d, w)
    end
end

function Mixture(distributions)
    return Mixture(distributions, fill(1 / length(distributions), length(distributions)))
end

const MultivariateNormalMixture = Mixture{<:MultivariateNormal}

@inline DensityInterface.DensityKind(::Mixture) = DensityInterface.HasDensity()

Base.length(mix::Mixture) = length(first(mix.distributions))
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

function squared_mean(d)
    μ = mean(d)
    return μ * transpose(μ)
end

second_moment(d) = cov(d)+ squared_mean(d)

function second_moment(mix::Mixture)
    d, w = distributions(mix), weights(mix)
    return sum(w[i] * second_moment(d[i]) for i in eachindex(d, w))
end

Statistics.cov(mix::Mixture) = second_moment(mix) - squared_mean(mix) 

function DensityInterface.densityof(mix::Mixture, x)
    d, w = distributions(mix), weights(mix)
    return sum(w[i] * densityof(d[i], x) for i in eachindex(d, w))
end

function DensityInterface.logdensityof(mix::Mixture, x)
    d, w = distributions(mix), weights(mix)
    return logsumexp(log(w[i]) + logdensityof(d[i], x) for i in eachindex(d, w))
end

function MultivariateNormal(mix::Mixture)
    return MultivariateNormal(mean(mix), cov(mix)) 
end
