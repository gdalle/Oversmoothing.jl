function plot_1d(
    csbm::CSBM, histograms::Vector{<:Matrix}, densities::Vector{<:Mixture}; layer::Integer
)
    (; sbm, features) = csbm
    (; S, Q) = sbm
    C = nb_communities(sbm)

    histograms_scalar = map(vec, histograms)
    joint_histogram_scalar = reduce(vcat, histograms_scalar)
    xrange = range(minimum(joint_histogram_scalar), maximum(joint_histogram_scalar), 200)

    fig = Figure()
    Label(
        fig[0, 1],
        """
        Contextual SBM with 2 communities - layer $layer
        N = $S       Q = $Q
        μ=$(only.(mean.(features)))     σ²=$(only.(cov.(features)))
        """;
        tellwidth=false,
    )
    ax = Axis(fig[1, 1]; ylabel="frequency")
    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)
    for c in 1:C
        hist!(
            ax,
            vec(histograms_scalar[c]);
            normalization=:pdf,
            bins=50,
            label="community $c",
            color=(colors[c], 0.5),
        )
        lines!(
            ax,
            xrange,
            [densityof(densities[c], [x]) for x in xrange];
            color=colors[c],
            linewidth=3,
        )
    end
    axislegend(ax)
    return fig
end
