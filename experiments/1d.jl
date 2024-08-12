using Pkg
Pkg.activate(dirname(@__DIR__))

using CairoMakie
using DensityInterface
using LaTeXStrings
using Latexify
using LinearAlgebra
using MonteCarloMeasurements
using OhMyThreads
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## Instance

sbm = SBM(100, 3, 0.02, 0.01)

features = [
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(0.03)),  #
    MultivariateNormal(SVector(-0.0), SMatrix{1,1}(0.01)),  #
    MultivariateNormal(SVector(+1.0), SMatrix{1,1}(0.02)),  #
]

csbm = CSBM(sbm, features)

## Computation

L = 4
histograms = @time embeddings(rng, csbm; nb_layers=L, nb_graphs=1000);
# densities1 = first_layer_mixtures(csbm; max_neighbors=50);
densities = random_walk_mixtures(rng, csbm; nb_layers=L, nb_graphs=100);

plot_1d(csbm, histograms, densities)

errors = random_walk_errors(rng, csbm; nb_layers=L, nb_graphs=100, nb_samples=100)

## Best depth

p_values = 10 .^ (-3:0.1:-1)
q_values = 10 .^ (-3:0.1:-1)

d_values = tmap(collect(Iterators.product(p_values, q_values))) do (p, q)
    q > p && return Particles([NaN])
    sbm = SBM(100, 2, p, q)
    features = [
        MultivariateNormal(SVector(-1.0), SMatrix{1,1}(2.0)),
        MultivariateNormal(SVector(+1.0), SMatrix{1,1}(2.0)),
    ]
    best_depth(rng, CSBM(sbm, features); nb_trajectories=10, nb_layers=5, nb_graphs=100)
end

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(
        fig[1, 1];
        title="Contextual SBM(p, q)",
        xlabel=L"p",
        ylabel=L"q",
        xscale=log10,
        yscale=log10,
    )
    hm = heatmap!(ax, p_values, q_values, pmean.(d_values); colormap=:plasma)
    Colorbar(fig[1, 2], hm; label="Optimal GCN depth")
    fig
end
