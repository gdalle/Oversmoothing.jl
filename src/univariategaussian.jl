struct UnivariateGaussian{T} <: AbstractMeasure
    μ::T
    σ²::T
end

function Base.show(io::IO, g::UnivariateGaussian)
    return print(io, "UnivariateGaussian($(g.μ)), $(g.σ²))")
end

Base.eltype(::UnivariateGaussian{T}) where {T} = T

Statistics.mean(g::UnivariateGaussian) = g.μ
Statistics.var(g::UnivariateGaussian) = g.σ²

function Random.rand(rng::AbstractRNG, g::UnivariateGaussian)
    z = randn(rng, eltype(g))
    return g.μ + sqrt(g.σ²) * z
end

function DensityInterface.logdensityof(g::UnivariateGaussian, x::Number)
    return normlogpdf(g.μ, sqrt(g.σ²), x)
end

function scale(g::UnivariateGaussian, λ::Number)
    return UnivariateGaussian(λ * g.μ, λ^2 * g.σ²)
end

function convolve(g1::UnivariateGaussian, g2::UnivariateGaussian)
    return UnivariateGaussian(g1.μ + g2.μ, g1.σ² + g2.σ²)
end
