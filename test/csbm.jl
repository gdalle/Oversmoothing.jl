using LinearAlgebra
using Oversmoothing
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

@testset "Linear CSBM 1d" begin
    csbm = LinearCSBM1d(; N=100, C=5, p_in=0.05, p_out=0.01, σ=0.1)
    (; features) = csbm
    for c in eachindex(features)
        @test mean(features[c]) == [c]
        @test cov(features[c]) ≈ [0.01;;]
    end
end

@testset "Circular CSBM 2d" begin
    csbm = CircularCSBM2d(; N=100, C=5, p_in=0.05, p_out=0.01, σ=0.1)
    (; features) = csbm
    for c in eachindex(features)
        @test norm(mean(features[c]) - mean(features[c % 5 + 1])) ≈ 1
    end
end
