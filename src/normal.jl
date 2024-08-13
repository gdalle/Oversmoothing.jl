struct MultivariateNormal{N,T<:Real,V<:SVector{N,T},M<:SMatrix{N,N,T},R<:LowerTriangular}
    μ::V
    Σ::M
    L::R
    Σ⁻¹::M
    logdetΣ::T
end

function MultivariateNormal(μ, Σ)
    @assert ishermitian(Σ)
    return MultivariateNormal(μ, Σ, cholesky(Σ).L, inv(Σ), logdet(Σ))
end

function UnivariateNormal(μ, σ²)
    return MultivariateNormal(SVector(μ), SMatrix{1,1}(σ²))
end

function BivariateNormal(μ, Σ)
    return MultivariateNormal(
        SVector{2}(μ[1], μ[2]), SMatrix{2,2}(Σ[1, 1], Σ[1, 2], Σ[2, 1], Σ[2, 2])
    )
end

@inline DensityInterface.DensityKind(::MultivariateNormal) = DensityInterface.HasDensity()

Base.length(::MultivariateNormal{N}) where {N} = N
Base.eltype(::MultivariateNormal{N,T}) where {N,T} = T

Statistics.mean(g::MultivariateNormal) = g.μ
Statistics.cov(g::MultivariateNormal) = g.Σ

rootcov(g::MultivariateNormal) = g.L
invcov(g::MultivariateNormal) = g.Σ⁻¹
logdetcov(g::MultivariateNormal) = g.logdetΣ

function Random.rand(rng::AbstractRNG, g::MultivariateNormal{N}) where {N}
    μ, L = mean(g), rootcov(g)
    z = @SVector(randn(rng, N))
    return μ .+ L * z
end

center(x::AbstractVector, μ::AbstractVector) = x - μ

function DensityInterface.logdensityof(
    g::MultivariateNormal{N}, x::AbstractVector
) where {N}
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    x̄ = center(x, μ)
    l = (-N * log2π - logdetΣ - dot(x̄, Σ⁻¹, x̄)) / 2
    return l
end

function DensityInterface.logdensityof(g::MultivariateNormal{1}, x::Number)
    μ, Σ⁻¹, logdetΣ = mean(g), invcov(g), logdetcov(g)
    l = (-log2π - logdetΣ - dot(x - only(μ), only(Σ⁻¹), x - only(μ))) / 2
    return l
end

function Base.isapprox(g1::MultivariateNormal, g2::MultivariateNormal; kwargs...)
    return isapprox(mean(g1), mean(g2); kwargs...) && isapprox(cov(g1), cov(g2); kwargs...)
end
