@kwdef struct Mixture{D,T}
    components::Vector{D}
    weights::T
end

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

function Statistics.var(mix::Mixture)
    c, w = components(mix), weights(mix)
    second_moment = sum(w[i] * (var(c[i]) + squared_mean(c[i])) for i in eachindex(c, w))
    return second_moment - squared_mean(mix)
end

function DensityInterface.logdensityof(mix::Mixture, x)
    c, w = components(mix), weights(mix)
    return logsumexp(log(w[i]) + logdensityof(c[i], x) for i in eachindex(c, w))
end
