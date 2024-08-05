using Random, Statistics
using CairoMakie
using Oversmoothing

rng = Random.default_rng()

sbm = SBM(3000, 3, 0.02, 0.01)

s, t = Oversmoothing.community_walk_probabilities(rng, sbm; nb_layers=2);
