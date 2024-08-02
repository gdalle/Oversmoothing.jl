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

function DensityInterface.logdensityof(g::MultivariateNormal, x::AbstractVector)
    k = length(g)
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    l = (-k * log2π - logdetΣ - dot(x - μ, Σ⁻¹, x - μ)) / 2
    @assert !isnan(l)
    return l
end

function DensityInterface.logdensityof(g::MultivariateNormal, x::Number)
    k = length(g)
    @assert k == 1
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    l = (-k * log2π - logdetΣ - dot(x - only(μ), only(Σ⁻¹), x - only(μ))) / 2
    @assert !isnan(l)
    return l
end

function StatsBase.entropy(g::MultivariateNormal)
    k = length(g)
    logdetΣ = logdetcov(g)
    return (k * (log2π + 1) + logdetΣ) / 2
end

function StatsBase.kldivergence(g0::MultivariateNormal, g1::MultivariateNormal)
    k = length(g0)
    μ0, Σ0, _, logdetΣ0 = mean(g0), cov(g0), invcov(g0), logdetcov(g0)
    μ1, _, Σ1⁻¹, logdetΣ1 = mean(g1), cov(g1), invcov(g1), logdetcov(g1)
    return (tr(Σ1⁻¹ * Σ0) + dot(μ1 - μ0, Σ1⁻¹, μ1 - μ0) - k + logdetΣ1 - logdetΣ0) / 2
end

function log_prod_norm(ga::MultivariateNormal, gb::MultivariateNormal)
    k = length(ga)
    μa, μb = mean(ga), mean(gb)
    Σa, Σb = cov(ga), cov(gb)
    return (-k * log2π - logdet(Σa + Σb) - dot(μb - μa, inv(Σa + Σb), μb - μa)) / 2
end

prod_norm(ga::MultivariateNormal, gb::MultivariateNormal) = exp(log_prod_norm(ga, gb))

function chernoff(g1::MultivariateNormal, g2::MultivariateNormal; α::Real=0.5)
    μ1, μ2 = mean(g1), mean(g2)
    Σ1, Σ2 = cov(g1), cov(g2)
    logdetΣ1, logdetΣ2 = logdetcov(g1), logdetcov(g2)
    Σ = (1 - α) * Σ1 + α * Σ2
    return (
        α * (1 - α) * dot(μ1 - μ2, inv(Σ), μ1 - μ2) +  #
        logdet(Σ) - (1 - α) * logdetΣ1 - α * logdetΣ2
    ) / 2
end

function Base.isapprox(g1::MultivariateNormal, g2::MultivariateNormal; kwargs...)
    return isapprox(mean(g1), mean(g2); kwargs...) && isapprox(cov(g1), cov(g2); kwargs...)
end
