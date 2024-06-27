using Oversmoothing
using LinearAlgebra
using StableRNGs
using Test

sbm = SBM(5, 2, 0.3, 0.1)
features = (MultivariateNormal([1.0], [0.5;;]), MultivariateNormal([2.0], [0.3;;]))

p = first_layer_mixtures(sbm, features);
