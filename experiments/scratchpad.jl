using Pkg
Pkg.activate(dirname(@__DIR__))

using CairoMakie
using Distributions
using Oversmoothing
using StableRNGs
using StaticArrays

rng = StableRNG(63)

graph = SBM(
    [30, 50, 20],
    [
        0.10 0.01 0.01
        0.01 0.10 0.01
        0.01 0.01 0.10
    ],
)

features = [
    MvNormal([+1.0, +1.0], [+1.0 +0.5; +0.5 +1.0]),
    MvNormal([-1.0, -1.0], [+1.0 +0.0; -0.0 +1.0]),
    MvNormal([+2.0, -2.0], [+1.0 -0.5; -0.5 +1.0]),
]

H = embeddings(rng, graph, features; layers=2)
H_split = split_by_community(H, graph)
