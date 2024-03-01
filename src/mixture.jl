
@kwdef struct Mixture{D,T} <: AbstractMeasure
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

function squared_mean(m::AbstractMeasure)
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

function Base.:+(mix1::Mixture, mix2::Mixture)
    c = vcat(components(mix1), components(mix2))
    w = vcat(weights(mix1), weights(mix2))
    return Mixture(c, w)
end

function scale(mix::Mixture, λ::Number)
    c = scale.(components(mix), Ref(λ))
    w = weights(mix)
    return Mixture(c, w)
end

function convolve(mix1::Mixture, mix2::Mixture)
    c1, w1 = components(mix1), weights(mix1)
    c2, w2 = components(mix2), weights(mix2)
    c = [convolve(c1[i], c2[j]) for i in eachindex(c1) for j in eachindex(c2)]
    w = [w1[i] * w2[j] for i in eachindex(w1) for j in eachindex(w2)]
    return Mixture(c, w)
end
