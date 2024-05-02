using CairoMakie
using Oversmoothing
using StableRNGs

rng = StableRNG(63)

function plot_1d_embeddings(H_samples, graph; kwargs...)
    C = nb_communities(graph)
    H_samples_by_community = split_by_community(graph, H_samples)

    fig = Figure()

    ax0 = Axis(fig[1, 1]; title="Embedding distribution of a GNN on the CSBM")
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
            label="community $c (size $(community_size(graph, c)))",
        )
    end
    axislegend(ax)
    return fig
end

# graph = SBM(; S=[10, 20], Q=[0.1 0.03; 0.03 0.1])
graph = ER(; N=10, q=0.1)
# features = [Gaussian(1.0, 0.02), Gaussian(-1.0, 0.02)]
features = [Gaussian(1.0, 0.1)]
contextual = Contextual(graph, features)
layers = 2

H_samples = embedding_samples(rng, contextual, NeighborhoodSum, layers)
H_samples_indep = embedding_samples_indep(rng, contextual, NeighborhoodSum, layers)

plot_1d_embeddings(H_samples, graph)
plot_1d_embeddings(H_samples_indep, graph)
