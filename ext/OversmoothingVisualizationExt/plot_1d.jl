function Oversmoothing.plot_1d_embeddings!(ax::Axis, sbm::SBM, H)
    C = nb_communities(sbm)
    H_split = split_by_community(H, sbm)
    for c in 1:C
        hist!(
            ax,
            vec(H_split[c]);
            normalization=:pdf,
            label="community $c (size $(community_size(sbm, c)))",
            bins=100,
        )
    end
    # axislegend(ax)
    return ax
end

function Oversmoothing.plot_1d_densities!(
    ax::Axis, sbm::SBM{C}, p::NTuple{C}; xmin=-1, xmax=1
) where {C}
    xrange = range(xmin, xmax, 200)
    for c in 1:C
        density_vals = [densityof(p[c], [x]) for x in xrange]
        lines!(
            ax, xrange, density_vals; label="community $c (size $(community_size(sbm, c)))"
        )
    end
    # axislegend(ax)
    return ax
end
