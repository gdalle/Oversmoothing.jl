using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using Distributions
using LinearAlgebra
using Oversmoothing
using Random: default_rng
using StableRNGs

BLAS.set_num_threads(1)
rng = default_rng()

function plot_emb_dens(graph, emb, dens)
    fig = Figure()
    ax1 = Axis(fig[1, 1]; title="Embedding histogram after $nb_layers layers")
    plot_1d_embeddings!(ax1, graph, emb)
    ax2 = Axis(fig[2, 1]; title="Approximate embedding density after $nb_layers layers")
    plot_1d_densities!(ax2, graph, dens; xmin=minimum(vec(emb)), xmax=maximum(vec(emb)))
    linkxaxes!(ax1, ax2)
    return fig
end

graph = SBM(200, 2, 0.02, 0.01)

features = [
    MvNormal([-1.0], [+0.1;;]),  #
    MvNormal([+1.0], [+0.1;;]),
]

nb_layers = 1

emb_history = @time embeddings(
    rng, graph, features; nb_layers, resample_graph=true, nb_graphs=1000
);
dens_history = @time state_evolution(graph, features; nb_layers);

plot_emb_dens(graph, last(emb_history), last(dens_history))
