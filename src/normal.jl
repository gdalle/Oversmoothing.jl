struct MultivariateNormal{V<:AbstractVector{<:Real},M<:AbstractMatrix{<:Real}}
    μ::V
    Σ::M
end

@inline DensityInterface.DensityKind(::MultivariateNormal) = DensityInterface.HasDensity()

Statistics.mean(g::MultivariateNormal) = g.μ
Statistics.cov(g::MultivariateNormal) = g.Σ

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
    μ, Σ = mean(g), cov(g)
    return (-k * log2π - logdet(Σ) - dot(x - μ, Σ \ (x - μ))) / 2
end

function StatsBase.entropy(g::MultivariateNormal)
    k = length(g)
    Σ = cov(g)
    return (k * (log2π + 1) + logdet(Σ)) / 2
end

function StatsBase.kldivergence(g0::MultivariateNormal, g1::MultivariateNormal)
    k = length(g0)
    μ0, Σ0 = mean(g0), cov(g0)
    μ1, Σ1 = mean(g1), cov(g1)
    return (tr(Σ1 \ Σ0) + dot(μ1 - μ0, Σ1 \ (μ1 - μ0)) - k + logdet(Σ1) - logdet(Σ0)) / 2
end

function log_prod_norm(ga::MultivariateNormal, gb::MultivariateNormal)
    k = length(ga)
    μa, μb = mean(ga), mean(gb)
    Σa, Σb = cov(ga), cov(gb)
    return (-k * log2π - logdet(Σa + Σb) - dot(μb - μa, (Σa + Σb) \ (μb - μa))) / 2
end

prod_norm(ga::MultivariateNormal, gb::MultivariateNormal) = exp(log_prod_norm(ga, gb))
