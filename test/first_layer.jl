using Oversmoothing
using LinearAlgebra
using StableRNGs
using Test

rng = StableRNG(63)

sbm = SBM(5, 2, 0.3, 0.1)
features = [MultivariateNormal([1.0], [0.5;;]), MultivariateNormal([2.0], [0.3;;])]
csbm = CSBM(sbm, features)

first_layer_mixtures(csbm);

embeddings(rng, csbm; nb_layers=3, nb_samples=10)
