struct Mixture{C,W}
    components::Vector{C}
    weights::Vector{W}
end

@inline DensityInterface.DensityKind(::Mixture) = DensityInterface.HasDensity()

function Base.show(io::IO, mix::Mixture{D}) where {D}
    return print(io, "Mixture of $(length(mix.components)) $D")
end

components(mix::Mixture) = mix.components
weights(mix::Mixture) = mix.weights

function Base.rand(rng::AbstractRNG, mix::Mixture)
    return rand(rng, sample(rng, components(mix), StatsBase.weights(weights(mix))))
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
