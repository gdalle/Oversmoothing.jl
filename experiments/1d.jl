using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using LinearAlgebra
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## Instance

sbm = SBM([100, 200], [0.05 0.01; 0.01 0.05])

features = (
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(+0.1)),  #
    MultivariateNormal(SVector(+1.0), SMatrix{1,1}(+0.2)),  #
)

## Computation

nb_layers = 6
emb_history = @time embeddings(
    rng, sbm, features; nb_layers, resample_graph=true, nb_graphs=100
);
dens_history = @time state_evolution(sbm, features; nb_layers, max_neighbors=50);

static_dens_history = make_static.(dens_history)

## KL

kl_history = @profview [
    kl_approx(static_dens_history[l, 1], static_dens_history[l, 2]) for
    l in axes(static_dens_history, 1)[1:2]
]

kl_history = @profview [
    kl_approx(dens_history[l, 1], dens_history[l, 2]) for
    l in axes(static_dens_history, 1)[1:2]
]

## Plotting

function plot_emb_dens(sbm, emb_history, dens_history, kl_history)
    (; S, Q) = sbm
    nb_layers = size(dens_history, 1) - 1
    xrange = range(minimum(emb_history), maximum(emb_history), 200)
    kl_mean_history = first.(kl_history)
    kl_std_history = last.(kl_history)

    C = nb_communities(sbm)
    emb = emb_history[:, 1, :, :]
    dens = dens_history[1, :]
    emb_split = split_by_community(emb, sbm)

    obs_title = Observable("Layer 0/$nb_layers")
    obs_emb_split = [Observable(vec(emb_split[c])) for c in 1:C]
    obs_dens_vals_split = [Observable([pdf(dens[c], [x]) for x in xrange]) for c in 1:C]
    dens_max = 1.05 * maximum(x -> maximum(x[]), obs_dens_vals_split)
    obs_limits = Observable((nothing, (0.0, dens_max)))
    obs_kl_mean_points = Observable(Point2f[(0, kl_mean_history[1])])

    fig = Figure()
    Label(
        fig[0, 1],
        """
Contextual SBM with 2 communities 
N = $S       Q = $Q
""";
        tellwidth=false,
    )
    ax1 = Axis(
        fig[1, 1];
        title=@lift("Embedding histogram - $($obs_title)"),
        limits=obs_limits,
        ylabel="frequency",
    )
    ax2 = Axis(
        fig[2, 1];
        title=@lift("Approximate embedding density - $($obs_title)"),
        limits=obs_limits,
        ylabel="frequency",
    )
    ax3 = Axis(
        fig[3, 1];
        title="Distance between communities",
        limits=((-1, nb_layers + 1), extrema(kl_mean_history)),
        xlabel="layer",
        ylabel="KL divergence",
    )
    linkxaxes!(ax1, ax2)
    for c in 1:C
        hist!(ax1, obs_emb_split[c]; normalization=:pdf, bins=20)
        lines!(ax2, xrange, obs_dens_vals_split[c];)
    end
    scatterlines!(ax3, obs_kl_mean_points; color=:black)

    record(fig, joinpath(@__DIR__, "oversmoothing.gif"), 0:nb_layers; framerate=1) do layer
        @info "Plotting layer $layer/$nb_layers"
        emb = emb_history[:, layer + 1, :, :]
        dens = dens_history[layer + 1, :]
        emb_split = split_by_community(emb, sbm)

        obs_title[] = "Layer $layer/$nb_layers"
        for c in 1:C
            obs_emb_split[c][] = vec(emb_split[c])
            obs_dens_vals_split[c][] = pdf(dens[c], transpose(xrange))
        end
        dens_max = 1.05 * maximum(x -> maximum(x[]), obs_dens_vals_split)
        obs_limits[] = (nothing, (0.0, dens_max))
        obs_kl_mean_points[] = push!(
            obs_kl_mean_points[], Point2f(layer, kl_mean_history[layer + 1])
        )
    end
end

plot_emb_dens(sbm, emb_history, dens_history, kl_history)
