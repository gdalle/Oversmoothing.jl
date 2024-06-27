using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using DensityInterface
using LinearAlgebra
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## Instance

sbm = SBM([30, 50], [0.03 0.01; 0.01 0.05])

features = (
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(0.01)),  #
    MultivariateNormal(SVector(+2.0), SMatrix{1,1}(0.03)),  #
)

## Computation

emb_history = @time embeddings(
    rng, sbm, features; nb_layers=1, resample_graph=true, nb_graphs=50
);
emb0 = emb_history[:, 1, :, :];
emb1 = emb_history[:, 2, :, :];
dens0 = [Mixture([features[1]], [1.0]), Mixture([features[2]], [1.0])]
dens1 = first_layer_mixtures(sbm, features; max_neighbors=20)

## Plotting

function plot_emb_dens(sbm, emb, dens)
    (; S, Q) = sbm
    C = nb_communities(sbm)
    emb_split = split_by_community(emb, sbm)
    xrange = range(minimum(emb), maximum(emb), 200)

    fig = Figure()
    Label(
        fig[0, 1], "Contextual SBM with 2 communities\nN = $S       Q = $Q"; tellwidth=false
    )
    ax1 = Axis(fig[1, 1]; title="Embedding histogram", ylabel="frequency")
    ax2 = Axis(
        fig[2, 1]; title="Approximate embedding density", ylabel="frequency"
    )
    linkxaxes!(ax1, ax2)
    for c in 1:C
        hist!(ax1, vec(emb_split[c]); normalization=:pdf, bins=50)
        lines!(ax2, xrange, [densityof(dens[c], [x]) for x in xrange])
    end
    return fig
end

plot_emb_dens(sbm, emb0, dens0)
plot_emb_dens(sbm, emb1, dens1)
