using Distributions
using Oversmoothing
using LinearAlgebra
using StableRNGs
using StaticArrays

graph = ER(10, 0.2)
features = [MvNormal(SVector(1.0), SMatrix{1,1}(0.5))]

π1 = state_evolution(graph, features; layers=2)
