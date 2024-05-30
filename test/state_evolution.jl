using Distributions
using Oversmoothing
using LinearAlgebra
using StableRNGs
using StaticArrays
using Test

## ER

er = ER(10, 0.2)
er_features = [MvNormal(SVector(1.0), SMatrix{1,1}(0.5))]

π_er = state_evolution(er, er_features; layers=2)
@test length(π_er) == 3

## SBM

sbm = SBM(10, 2, 0.3, 0.1)
sbm_features = [
    MvNormal(SVector(1.0), SMatrix{1,1}(0.5)), MvNormal(SVector(2.0), SMatrix{1,1}(0.3))
]

π_sbm = state_evolution(sbm, sbm_features; layers=3)
@test length(π_sbm) == 4
