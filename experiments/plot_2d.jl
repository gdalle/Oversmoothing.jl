using CairoMakie
using Oversmoothing
using StableRNGs

rng = StableRNG(63)

function plot_2d_embeddings(H_samples, contextual)
    C = nb_communities(contextual)
    H_samples_by_community = split_by_community(contextual, H_samples)

    fig = Figure()

    ax0 = Axis(
        fig[1, 1:C];
        aspect=1,
        title="Embedding distribution for a $layers-layer GCN on the CSBM",
        xlabel="all communities",
    )
    hb0 = hexbin!(
        ax0,
        first.(reduce(vcat, H_samples_by_community)),
        last.(reduce(vcat, H_samples_by_community));
        bins=40,
    )
    for c in 1:C
        ax = Axis(
            fig[2, c];
            aspect=1,
            limits=ax0.limits,
            xlabel="community $c (size $(community_size(contextual, c)))",
        )
        hb = hexbin!(
            ax, first.(H_samples_by_community[c]), last.(H_samples_by_community[c]); bins=40
        )
        linkxaxes!(ax, ax0)
        linkyaxes!(ax, ax0)
    end

    resize_to_layout!(fig)
    return fig
end

graph = SBM(
    [30, 50, 20],
    [
        0.10 0.01 0.01
        0.01 0.10 0.01
        0.01 0.01 0.10
    ],
)
features = [
    MultivariateGaussian([+1.0, +1.0], [+1.0 +0.5; +0.5 +1.0]),
    MultivariateGaussian([-1.0, -1.0], [+1.0 +0.0; -0.0 +1.0]),
    MultivariateGaussian([+2.0, -2.0], [+1.0 -0.5; -0.5 +1.0]),
]
contextual = Contextual(graph, features)
convolution = NeighborhoodAverage()
layers = 1
plot_2d_embeddings(rng, contextual, convolution, layers; samples=100)
