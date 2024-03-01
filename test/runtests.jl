using Oversmoothing
using Test

@testset "Oversmoothing.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(Oversmoothing; ambiguities=false, deps_compat=false)
    end
    @testset "JuliaFormatter" begin
        @test JuliaFormatter.format(dirname(@__DIR__), overwrite=false)
    end
    @testset "JET" begin
        JET.test_package(Oversmoothing; target_defined_modules=true)
    end
end
