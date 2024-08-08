struct MultivariateNormal{T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    μ::V
    Σ::M
    Σ⁻¹::M
    logdetΣ::T
end

MultivariateNormal(μ, Σ) = MultivariateNormal(μ, Σ, inv(Σ), logdet(Σ))

@inline DensityInterface.DensityKind(::MultivariateNormal) = DensityInterface.HasDensity()

Statistics.mean(g::MultivariateNormal) = g.μ
Statistics.cov(g::MultivariateNormal) = g.Σ
invcov(g::MultivariateNormal) = g.Σ⁻¹
logdetcov(g::MultivariateNormal) = g.logdetΣ

Base.length(g::MultivariateNormal) = length(mean(g))
Base.eltype(g::MultivariateNormal) = promote_type(eltype(mean(g)), eltype(cov(g)))

function Random.rand(
    rng::AbstractRNG, g::MultivariateNormal, dims::Vararg{Integer,N}
) where {N}
    k = length(g)
    μ, Σ = mean(g), cov(g)
    L = cholesky(Σ).L
    z = randn(rng, k, dims...)
    return μ .+ L * z
end

center(x::AbstractVector, μ::AbstractVector) = x - μ

function DensityInterface.logdensityof(g::MultivariateNormal, x::AbstractVector)
    k = length(g)
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    x̄ = center(x, μ)
    l = (-k * log2π - logdetΣ - dot(x̄, Σ⁻¹, x̄)) / 2
    return l
end

function DensityInterface.logdensityof(g::MultivariateNormal, x::Number)
    k = length(g)
    if k > 1
        throw(ArgumentError("Distribution is not scalar-valued"))
    end
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    l = (-k * log2π - logdetΣ - dot(x - only(μ), only(Σ⁻¹), x - only(μ))) / 2
    return l
end

function Base.isapprox(g1::MultivariateNormal, g2::MultivariateNormal; kwargs...)
    return isapprox(mean(g1), mean(g2); kwargs...) && isapprox(cov(g1), cov(g2); kwargs...)
end
