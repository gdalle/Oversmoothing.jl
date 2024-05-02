using Pkg
Pkg.activate(dirname(@__DIR__))

using Distributions
using Oversmoothing
using StableRNGs

rng = StableRNG(63)

graph = SBM(
    [30, 50, 20],
    [
        0.05 0.01 0.01
        0.01 0.05 0.01
        0.01 0.01 0.05
    ],
)

features = [
    MvNormal([+1.0, +1.0], [+1.0 +0.5; +0.5 +1.0]),
    MvNormal([-1.0, -1.0], [+1.0 +0.0; -0.0 +1.0]),
    MvNormal([+2.0, -2.0], [+1.0 -0.5; -0.5 +1.0]),
]

H = @time embeddings_(rng, graph, features; layers=3, resample_graph=false);
H_split = split_by_community(H, graph)

plot_2d_embeddings(H_split)
