struct MultivariateNormal{
    T<:Real,V<:AbstractVector{T},M<:AbstractMatrix{T},R<:AbstractMatrix{T}
}
    μ::V
    Σ::M
    L::R
    Σ⁻¹::M
    logdetΣ::T
end

function MultivariateNormal(μ::AbstractVector, Σ::AbstractMatrix)
    @assert ishermitian(Σ)
    return MultivariateNormal(μ, Σ, cholesky(Σ).L, inv(Σ), logdet(Σ))
end

function UnivariateNormal(μ::Real, σ²::Real)
    return MultivariateNormal(SVector(μ), SMatrix{1,1}(σ²))
end

function BivariateNormal(μ::AbstractVector, Σ::AbstractMatrix)
    return MultivariateNormal(
        SVector{2}(μ[1], μ[2]), SMatrix{2,2}(Σ[1, 1], Σ[1, 2], Σ[2, 1], Σ[2, 2])
    )
end

@inline DensityInterface.DensityKind(::MultivariateNormal) = DensityInterface.HasDensity()

Base.length(dist::MultivariateNormal) = length(dist.μ)
Base.eltype(::MultivariateNormal{T}) where {T} = T

Statistics.mean(g::MultivariateNormal) = g.μ
Statistics.cov(g::MultivariateNormal) = g.Σ

rootcov(g::MultivariateNormal) = g.L
invcov(g::MultivariateNormal) = g.Σ⁻¹
logdetcov(g::MultivariateNormal) = g.logdetΣ

function Random.rand(
    rng::AbstractRNG, g::MultivariateNormal{T,<:SVector{N,T},<:SMatrix{N,N,T}}
) where {N,T}
    μ, L = mean(g), rootcov(g)
    z = @SVector(randn(rng, T, N))
    return μ + L * z
end

function Random.rand(rng::AbstractRNG, g::MultivariateNormal{T}, n::Integer) where {T}
    return [rand(rng, g) for _ in 1:n]
end

function DensityInterface.logdensityof(g::MultivariateNormal, x::AbstractVector)
    N = length(g)
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    x̄ = x - μ
    l = (-N * log2π - logdetΣ - dot(x̄, Σ⁻¹, x̄)) / 2
    return l
end

function DensityInterface.logdensityof(g::MultivariateNormal, x::Number)
    @assert length(g) == 1
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    l = (-log2π - logdetΣ - dot(x - only(μ), only(Σ⁻¹), x - only(μ))) / 2
    return l
end
