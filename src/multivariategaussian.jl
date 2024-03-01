
struct MultivariateGaussian{T,V<:AbstractVector{T},M<:AbstractMatrix{T}} <: AbstractMeasure
    μ::V
    Σ::M
    chol::Cholesky{T,M}
    function MultivariateGaussian(
        μ::V, Σ::M
    ) where {T,V<:AbstractVector{T},M<:AbstractMatrix{T}}
        return new{T,V,M}(μ, Σ, cholesky(Σ))
    end
end

function Base.show(io::IO, g::MultivariateGaussian)
    return print(io, "MultivariateGaussian($(g.μ)), $(g.Σ))")
end

Base.eltype(::MultivariateGaussian{T}) where {T} = T
Base.length(g::MultivariateGaussian) = length(g.μ)

Statistics.mean(g::MultivariateGaussian) = g.μ
Statistics.var(g::MultivariateGaussian) = g.Σ

function Random.rand(rng::AbstractRNG, g::MultivariateGaussian)
    z = randn(rng, eltype(g), length(g))
    return g.μ + g.chol.L * z
end

function DensityInterface.logdensityof(g::MultivariateGaussian, x::AbstractVector)
    return (
        -(length(g) * log2π / 2) - logdet(g.chol) / 2 - dot(x - g.μ, g.chol \ (x - g.μ)) / 2
    )
end

function scale(g::MultivariateGaussian, λ::Number)
    return MultivariateGaussian(λ * g.μ, λ^2 * g.Σ)
end

function convolve(g1::MultivariateGaussian, g2::MultivariateGaussian)
    return MultivariateGaussian(g1.μ + g2.μ, g1.Σ + g2.Σ)
end
