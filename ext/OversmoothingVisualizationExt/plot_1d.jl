function Oversmoothing.plot_1d_embeddings!(
    ax::Axis, graph::AbstractRandomGraph, H::AbstractArray
)
    C = nb_communities(graph)
    H_split = split_by_community(H, graph)
    for c in 1:C
        hist!(
            ax,
            vec(H_split[c]);
            normalization=:pdf,
            label="community $c (size $(community_size(graph, c)))",
            bins=50,
        )
    end
    axislegend(ax)
    return ax
end

function Oversmoothing.plot_1d_densities!(
    ax::Axis, graph::AbstractRandomGraph, π::NTuple{C,<:Mixture}; xmin=-1, xmax=1
) where {C}
    xrange = range(xmin, xmax, 100)
    for c in 1:C
        density_vals = [densityof(π[c], [x]) for x in xrange]
        lines!(
            ax,
            xrange,
            density_vals;
            label="community $c (size $(community_size(graph, c)))",
        )
    end
    axislegend(ax)
    return ax
end
