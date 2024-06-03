using Distributions
using Oversmoothing
using LinearAlgebra
using StableRNGs
using StaticArrays
using Test

sbm = SBM(10, 2, 0.3, 0.1)
features = (
    MvNormal(SVector(1.0), SMatrix{1,1}(0.5)), MvNormal(SVector(2.0), SMatrix{1,1}(0.3))
)

p = state_evolution(sbm, features; nb_layers=3)
@test length(p) == 2
@test length(p[1]) == 4
