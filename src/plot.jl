function plot_1d(
    csbm::CSBM, histograms::Vector{<:Matrix}, densities::Vector{<:Mixture}; layer::Integer
)
    (; sbm, features) = csbm
    (; S, Q) = sbm
    C = nb_communities(sbm)

    joint_histogram_scalar = mapreduce(vec, vcat, histograms)
    xrange = range(extrema(joint_histogram_scalar)..., 200)

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    fig = Figure()
    Label(
        fig[0, 1],
        """
        Contextual SBM with $C communities - layer $layer
        N = $S       Q = $Q
        μ=$(only.(mean.(features)))     σ²=$(only.(cov.(features)))
        """;
        tellwidth=false,
    )
    ax = Axis(fig[1, 1]; ylabel="frequency")
    for c in 1:C
        hist!(
            ax,
            vec(histograms[c]);
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

function plot_2d(
    csbm::CSBM, histograms::Vector{<:Matrix}, densities::Vector{<:Mixture}; layer::Integer
)
    (; sbm, features) = csbm
    (; S, Q) = sbm
    C = nb_communities(sbm)

    joint_histogram_x = mapreduce(hist -> hist[:, 1], vcat, histograms)
    joint_histogram_y = mapreduce(hist -> hist[:, 2], vcat, histograms)
    xrange = range(extrema(joint_histogram_x)..., 100)
    yrange = range(extrema(joint_histogram_y)..., 100)
    zs = SVector.(xrange, yrange')
    ls = [logdensityof.(Ref(densities[c]), zs) for c in 1:C]
    lmin, lmax = extrema(mapreduce(vec, vcat, ls))

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    fig = Figure(; size=(1000, 500))
    Label(
        fig[0, 1:C],
        """
        Contextual SBM with $C communities - layer $layer
        N = $S       Q = $Q
        μ=$(Vector.(mean.(features)))
        Σ=$(Matrix.(cov.(features)))
        """;
        tellwidth=false,
    )

    axes = [Axis(fig[1, c]; aspect=1) for c in 1:C]
    for c in 2:C
        linkxaxes!(axes[1], axes[c])
        linkyaxes!(axes[1], axes[c])
    end

    for c in 1:C
        ls = logdensityof.(Ref(densities[c]), zs)
        scatter!(
            axes[c],
            histograms[c][:, 1],
            histograms[c][:, 2];
            color=colors[c],
            label="community $c",
            alpha=1 / 10,
        )
        contour!(
            axes[c], xrange, yrange, ls; colorrange=(lmin, lmax), levels=15, linewidth=2
        )
        axislegend(axes[c])
    end
    Colorbar(
        fig[2, 1:C]; limits=(lmin, lmax), vertical=false, label="Mixture loglikelihood"
    )

    colsize!(fig.layout, 1, Aspect(1, 1))
    return fig
end
