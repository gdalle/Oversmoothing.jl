using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using DensityInterface
using Distributions
using LinearAlgebra
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

sbm = SBM([300, 700], [0.03 0.002;0.002 0.01])

features = (
    MvNormal(SVector(-1.0), SMatrix{1,1}(+0.01)),  #
    MvNormal(SVector(+1.0), SMatrix{1,1}(+0.03)),  #
)

nb_layers = 8
emb_history = @time embeddings(
    rng, sbm, features; nb_layers, resample_graph=true, nb_graphs=10
);
dens_history = @time state_evolution(sbm, features; nb_layers, max_neighbors=100);

plot_emb_dens(sbm, emb_history, dens_history)

function plot_emb_dens(sbm, emb_history, dens_history)
    (; S, Q) = sbm
    nb_layers = size(dens_history, 1) - 1
    xrange = range(minimum(emb_history), maximum(emb_history), 200)

    C = nb_communities(sbm)
    emb = emb_history[:, 1, :, :]
    dens = dens_history[1, :]
    emb_split = split_by_community(emb, sbm)

    obs_title = Observable("Layer 0/$nb_layers")
    obs_emb_split = [Observable(vec(emb_split[c])) for c in 1:C]
    obs_dens_vals_split = [
        Observable([densityof(dens[c], [x]) for x in xrange]) for c in 1:C
    ]
    dens_max = 1.05 * maximum(x -> maximum(x[]), obs_dens_vals_split)
    obs_limits = Observable((nothing, (0.0, dens_max)))

    fig = Figure()
    Label(
        fig[1, 1],
        """
Contextual SBM with 2 communities
N = $S       Q = $Q
""";
        tellwidth=false,
    )
    ax1 = Axis(
        fig[2, 1]; title=@lift("Embedding histogram - $($obs_title)"), limits=obs_limits
    )
    ax2 = Axis(
        fig[3, 1];
        title=@lift("Approximate embedding density - $($obs_title)"),
        limits=obs_limits,
    )
    linkxaxes!(ax1, ax2)
    for c in 1:C
        hist!(ax1, obs_emb_split[c]; normalization=:pdf, bins=50)
        lines!(ax2, xrange, obs_dens_vals_split[c];)
    end

    record(fig, joinpath(@__DIR__, "oversmoothing.gif"), 0:nb_layers; framerate=2) do layer
        @info "layer $layer/$nb_layers"
        emb = emb_history[:, layer + 1, :, :]
        dens = dens_history[layer + 1, :]
        emb_split = split_by_community(emb, sbm)

        obs_title[] = "Layer $layer/$nb_layers"
        for c in 1:C
            obs_emb_split[c][] = vec(emb_split[c])
            obs_dens_vals_split[c][] = [densityof(dens[c], [x]) for x in xrange]
        end
        dens_max = 1.05 * maximum(x -> maximum(x[]), obs_dens_vals_split)
        obs_limits[] = (nothing, (0.0, dens_max))
    end
end
