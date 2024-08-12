using LinearAlgebra
using Oversmoothing
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

sbm = SBM(99, 3, 0.002, 0.001)

features = [
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(0.03)),  #
    MultivariateNormal(SVector(0.0), SMatrix{1,1}(0.01)),  #
    MultivariateNormal(SVector(+1.0), SMatrix{1,1}(0.02)),  #
]

csbm = CSBM(sbm, features)

random_walk_errors(rng, csbm; nb_layers=L, nb_graphs=100, nb_samples=100)
