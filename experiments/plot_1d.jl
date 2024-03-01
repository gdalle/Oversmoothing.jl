using CairoMakie
using Oversmoothing
using StableRNGs

rng = StableRNG(63)

function plot_1d_embeddings(
    rng::AbstractRNG,
    contextual::Contextual,
    convolution::AbstractConvolution,
    layers::Integer;
    kwargs...,
)
    C = nb_communities(contextual)
    H_samples = embedding_samples(rng, contextual, convolution, layers; kwargs...)
    H_samples_by_community = split_by_community(contextual, H_samples)

    fig = Figure()

    ax0 = Axis(
        fig[1, 1];
        title="Embedding distribution after $layers convolutional layers on the CSBM",
    )
    hist!(
        ax0,
        reduce(vcat, H_samples_by_community);
        normalization=:pdf,
        bins=100,
        color=:black,
        label="all communities",
    )
    axislegend(ax0)

    ax = Axis(fig[2, 1])
    linkxaxes!(ax, ax0)
    for c in 1:C
        hist!(
            ax,
            H_samples_by_community[c];
            normalization=:pdf,
            bins=100,
            label="community $c (size $(community_size(contextual, c)))",
        )
    end
    axislegend(ax)
    return fig
end

graph = SBM(; S=[10, 20], Q=[0.1 0.03; 0.03 0.1])
features = [Gaussian(1.0, 0.02), Gaussian(-1.0, 0.02)]
contextual = Contextual(graph, features)
convolution = NeighborhoodAverage()
layers = 1
plot_1d_embeddings(rng, contextual, convolution, layers; samples=100)
