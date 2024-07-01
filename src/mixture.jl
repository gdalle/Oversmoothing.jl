struct Mixture{C,W}
    components::Vector{C}
    weights::Vector{W}

    function Mixture(c::Vector{C}, w::Vector{W}) where {C,W}
        @assert sum(w) ≈ one(W)
        @assert length(c) == length(w)
        return new{C,W}(c, w)
    end
end

const MultivariateNormalMixture = Mixture{<:MultivariateNormal}

@inline DensityInterface.DensityKind(::Mixture) = DensityInterface.HasDensity()

function Base.show(io::IO, mix::Mixture{D}) where {D}
    return print(io, "Mixture of $(length(mix.components)) $D")
end

Base.length(mix::Mixture) = length(mix.components)
components(mix::Mixture) = mix.components
weights(mix::Mixture) = mix.weights

function Base.rand(rng::AbstractRNG, mix::Mixture)
    ci = sample(rng, components(mix), StatsBase.weights(weights(mix)))
    return rand(rng, ci)
end

function Base.rand(rng::AbstractRNG, mix::Mixture, n::Integer)
    cis = sample(rng, components(mix), StatsBase.weights(weights(mix)), n)
    return [rand(rng, ci) for ci in cis]
end

function Statistics.mean(mix::Mixture)
    c, w = components(mix), weights(mix)
    return sum(w[i] * mean(c[i]) for i in eachindex(c, w))
end

function squared_mean(m)
    μ = mean(m)
    return μ * transpose(μ)
end

function Statistics.cov(mix::Mixture)
    c, w = components(mix), weights(mix)
    second_moment = sum(w[i] * (cov(c[i]) + squared_mean(c[i])) for i in eachindex(c, w))
    Σ = second_moment - squared_mean(mix)
    return Σ
end

function DensityInterface.densityof(mix::Mixture, x)
    c, w = components(mix), weights(mix)
    return sum(w[i] * densityof(c[i], x) for i in eachindex(c, w))
end

function DensityInterface.logdensityof(mix::Mixture, x)
    c, w = components(mix), weights(mix)
    return logsumexp(log(w[i]) + logdensityof(c[i], x) for i in eachindex(c, w))
end
