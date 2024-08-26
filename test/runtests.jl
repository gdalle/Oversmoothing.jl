using Aqua
using JuliaFormatter
using JET
using Oversmoothing
using Test

@testset verbose = true "Oversmoothing.jl" begin
    @testset "Formalities" begin
        Aqua.test_all(Oversmoothing; ambiguities=false, deps_compat=false, stale_deps=false)
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
    @testset "Depth" begin
        include("depth.jl")
    end
end
