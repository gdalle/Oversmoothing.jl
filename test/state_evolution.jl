using Distributions
using Oversmoothing
using LinearAlgebra
using StableRNGs
using Test

sbm = SBM(10, 2, 0.3, 0.1)
features = (MvNormal([1.0], [0.5;;]), MvNormal([2.0], [0.3;;]))

p = state_evolution(sbm, features; nb_layers=3)
@test size(p) == (4, 2)
