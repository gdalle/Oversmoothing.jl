using Pkg
Pkg.activate(@__DIR__)

using CairoMakie
using Colors
using DensityInterface
using LinearAlgebra
using OhMyThreads
using Oversmoothing
using Random: default_rng
using StableRNGs
using StaticArrays

BLAS.set_num_threads(1)
rng = default_rng()

## Instance

sbm = SBM(100, 2, 0.10, 0.01)

features = (
    MultivariateNormal(SVector(-1.0), SMatrix{1,1}(0.5)),  #
    MultivariateNormal(SVector(+1.0), SMatrix{1,1}(0.5)),  #
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
    ax = Axis(fig[1, 1]; ylabel="frequency")
    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)
    for c in 1:C
        hist!(
            ax,
            vec(emb_split[c]);
            normalization=:pdf,
            bins=50,
            label="community $c",
            color=(colors[c], 0.5),
        )
        lines!(
            ax,
            xrange,
            [densityof(dens[c], [x]) for x in xrange];
            color=colors[c],
            linewidth=3,
        )
    end
    axislegend(ax)
    return fig
end

plot_emb_dens(sbm, emb0, dens0)
plot_emb_dens(sbm, emb1, dens1)

## Error

bayes_classification_error_interval(
    rng, dens0, sbm.S ./ sum(sbm.S); nb_samples=1000, nb_errors=10
)
bayes_classification_error_interval(
    rng, dens1, sbm.S ./ sum(sbm.S); nb_samples=1000, nb_errors=10
)

## Several

logp_values = collect(-1.5:0.1:-1)
logq_values = collect(-2:0.1:-1.5)

logpq_indices = collect(Iterators.product(eachindex(logp_values), eachindex(logq_values)))

error_behavior = tmap(logpq_indices) do (a, b)
    p, q = 10.0 .^ logp_values[a], 10.0 .^ logq_values[b]
    sbm = SBM(100, 2, p, q)
    dens1 = first_layer_mixtures(sbm, features; max_neighbors=20)

    e0⁻, e0⁺ = bayes_classification_error_interval(
        rng, dens0, sbm.S ./ sum(sbm.S); nb_samples=1000, nb_errors=10
    )
    e1⁻, e1⁺ = bayes_classification_error_interval(
        rng, dens1, sbm.S ./ sum(sbm.S); nb_samples=1000, nb_errors=10
    )
    if e1⁻ > e0⁺  # error increase
        return e1⁻ - e0⁺
    elseif e1⁺ < e0⁻
        return -(e0⁻ - e1⁺)
    else
        return missing
    end
end

error_behavior

fig = Figure()
ax = Axis(
    fig[1, 1];
    title="Guaranteed error evolution after one convolution layer",
    xlabel="log p",
    ylabel="log q",
    ygridstyle=nothing,
)
hidexdecorations!(ax; label=false, ticklabels=false)
hideydecorations!(ax; label=false, ticklabels=false)
hm = heatmap!(
    ax,
    logp_values,
    logq_values,
    error_behavior;
    colorrange=(
        -maximum(abs, skipmissing(error_behavior)),
        maximum(abs, skipmissing(error_behavior)),
    ),
    colormap=:balance,
)
Colorbar(fig[1, 2], hm)
fig
