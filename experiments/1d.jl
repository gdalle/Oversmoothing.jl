using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using Distributions
using LinearAlgebra
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

function plot_emb_dens(graph, emb, dens; layer)
    fig = Figure()
    ax1 = Axis(fig[1, 1]; title="Embedding histogram at layer $layer")
    plot_1d_embeddings!(ax1, graph, emb)
    ax2 = Axis(fig[2, 1]; title="Approximate embedding density at layer $layer")
    plot_1d_densities!(ax2, graph, dens; xmin=minimum(vec(emb)), xmax=maximum(vec(emb)))
    linkxaxes!(ax1, ax2)
    return fig
end

graph = SBM(1000, 2, 0.01, 0.002)

features = [
    MvNormal(SVector(-2.0), SMatrix{1,1}(+0.01)),  #
    MvNormal(SVector(+1.0), SMatrix{1,1}(+0.02)),  #
]

nb_layers = 10
emb_history = @time embeddings(
    rng, graph, features; nb_layers, resample_graph=true, nb_graphs=100
);
dens_history = @time state_evolution(graph, features; nb_layers, max_neighbors=100);

layer = 2
plot_emb_dens(graph, emb_history[layer], dens_history[layer]; layer)
