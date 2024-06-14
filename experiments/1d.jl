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

sbm = SBM(1000, 2, 0.04, 0.01)

features = (
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(0.01)),  #
    MultivariateNormal(SVector(+2.0), SMatrix{1,1}(0.03)),  #
)

## Computation

nb_layers = 8
emb_history = @time embeddings(
    rng, sbm, features; nb_layers, resample_graph=true, nb_graphs=50
);
dens_history = @time state_evolution(sbm, features; nb_layers, max_neighbors=50);

## KL

# kl_lowerbound_history = [
#     kl_lowerbound(dens_history[l, 1], dens_history[l, 2]) for l in axes(dens_history, 1)
# ]

# kl_upperbound_history = [
#     kl_upperbound(dens_history[l, 1], dens_history[l, 2]) for l in axes(dens_history, 1)
# ]

# diff((kl_lowerbound_history .+ kl_upperbound_history) ./ 2)

## Plotting

function plot_emb_dens(
    sbm,
    emb_history,
    dens_history,
    kl_lowerbound_history=nothing,
    kl_upperbound_history=nothing,
)
    (; S, Q) = sbm
    nb_layers = size(dens_history, 1) - 1

    C = nb_communities(sbm)
    emb = emb_history[:, 1, :, :]
    dens = dens_history[1, :]
    emb_split = split_by_community(emb, sbm)
    xrange = range(minimum(emb), maximum(emb), 200)

    obs_title = Observable("Layer 0/$nb_layers")
    obs_emb_split = [Observable(vec(emb_split[c])) for c in 1:C]
    obs_dens_vals_split_points = [
        Observable(Point2f[(x, densityof(dens[c], [x])) for x in xrange]) for c in 1:C
    ]
    # obs_kl_lowerbound_points = Observable(Point2f[(0, kl_lowerbound_history[1])])
    # obs_kl_upperbound_points = Observable(Point2f[(0, kl_upperbound_history[1])])

    fig = Figure()
    Label(
        fig[0, 1], "Contextual SBM with 2 communities\nN = $S       Q = $Q"; tellwidth=false
    )
    ax1 = Axis(
        fig[1, 1]; title=@lift("Embedding histogram - $($obs_title)"), ylabel="frequency"
    )
    ax2 = Axis(
        fig[2, 1];
        title=@lift("Approximate embedding density - $($obs_title)"),
        ylabel="frequency",
    )
    # ax3 = Axis(
    #     fig[3, 1];
    #     title="Distance between communities",
    #     xlabel="layer",
    #     ylabel="KL divergence",
    # )
    linkxaxes!(ax1, ax2)
    for c in 1:C
        hist!(ax1, obs_emb_split[c]; normalization=:pdf, bins=50)
        lines!(ax2, obs_dens_vals_split_points[c];)
    end
    # scatterlines!(ax3, obs_kl_lowerbound_points; color=:black)
    # scatterlines!(ax3, obs_kl_upperbound_points; color=:black, marker=:rect)

    record(fig, joinpath(@__DIR__, "oversmoothing.gif"), 0:nb_layers; framerate=1) do layer
        @info "Plotting layer $layer/$nb_layers"
        emb = emb_history[:, layer + 1, :, :]
        dens = dens_history[layer + 1, :]
        emb_split = split_by_community(emb, sbm)
        xrange = range(minimum(emb), maximum(emb), 200)

        obs_title[] = "Layer $layer/$nb_layers"
        for c in 1:C
            obs_emb_split[c][] = vec(emb_split[c])
            obs_dens_vals_split_points[c][] = [
                Point2f(x, densityof(dens[c], [x])) for x in xrange
            ]
        end
        # obs_kl_lowerbound_points[] = push!(
        #     obs_kl_lowerbound_points[], Point2f(layer, kl_lowerbound_history[layer + 1])
        # )
        # obs_kl_upperbound_points[] = push!(
        #     obs_kl_upperbound_points[], Point2f(layer, kl_upperbound_history[layer + 1])
        # )
        autolimits!(ax1)
        autolimits!(ax2)
        # autolimits!(ax3)
    end
end

plot_emb_dens(sbm, emb_history, dens_history)
