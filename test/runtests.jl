using Aqua
using JuliaFormatter
using JET
using Oversmoothing
using Test

@testset verbose = true "Oversmoothing.jl" begin
    @testset "Formalities" begin
        Aqua.test_all(Oversmoothing; ambiguities=false, deps_compat=false, stale_deps=false)
        @test JuliaFormatter.format(Oversmoothing, overwrite=true)
        JET.test_package(Oversmoothing; target_defined_modules=true)
    end

    @testset "Normal" begin
        include("normal.jl")
    end
    @testset "Mixture" begin
        include("mixture.jl")
    end
    @testset "Stochastic Block Model" begin
        include("sbm.jl")
    end
    @testset "First layer" begin end
    @testset "Random walk" begin
        include("random_walk.jl")
    end
end
