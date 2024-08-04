using Random, Statistics
using CairoMakie
using Oversmoothing

rng = Random.default_rng()

sbm = SBM(3000, 3, 0.02, 0.01)

r, s = Oversmoothing.community_walk_probabilities(rng, sbm; nb_layers=2);

fig = Figure()
for c0 in 1:nb_communities(sbm), c1 in 1:nb_communities(sbm)
    scatter!(Axis(fig[c0, c1]), r[c0, c1], s[c0, c1]; alpha=0.1)
end
fig
