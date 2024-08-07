function plot_1d(
    csbm::CSBM,
    histograms::Vector{<:Vector{<:Matrix}},
    densities::Vector{<:Vector{<:Mixture}};
)
    (; sbm) = csbm
    C = nb_communities(sbm)

    joint_histogram_scalar = mapreduce(vec, vcat, reduce(vcat, histograms))
    xrange = range(extrema(joint_histogram_scalar)..., 200)

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    fig = Figure(; size=(600, 200 * length(histograms)))
    axes = Axis[]
    Label(fig[0, 1], "Contextual SBM in 1D"; tellwidth=false)
    for l in eachindex(histograms, densities)
        ax = Axis(fig[l, 1]; title="Layer $(l-1)")
        push!(axes, ax)
        linkxaxes!(ax, axes[1])
        for c in 1:C
            hist!(
                ax,
                vec(histograms[l][c]);
                normalization=:pdf,
                bins=50,
                label="community $c",
                color=(colors[c], 0.5),
            )
            lines!(
                ax,
                xrange,
                [densityof(densities[l][c], [x]) for x in xrange];
                color=colors[c],
                linewidth=2,
            )
        end
    end
    return fig
end

function plot_2d(
    csbm::CSBM,
    histograms::Vector{<:Vector{<:Matrix}},
    densities::Vector{<:Vector{<:Mixture}};
)
    (; sbm) = csbm
    C = nb_communities(sbm)
    L = length(histograms)

    joint_histogram_x = mapreduce(hist -> hist[:, 1], vcat, reduce(vcat, histograms))
    joint_histogram_y = mapreduce(hist -> hist[:, 2], vcat, reduce(vcat, histograms))
    xrange = range(extrema(joint_histogram_x)..., 100)
    yrange = range(extrema(joint_histogram_y)..., 100)
    zs = SVector.(xrange, yrange')
    ls = [logdensityof.(Ref(densities[l][c]), zs) for l in 1:L, c in 1:C]

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    fig = Figure(; size=(700, 200 * L))
    all_axes = Axis[]
    Label(fig[-1, 1:C], "Contextual SBM in 2D"; tellwidth=false, fontsize=20)
    for c in 1:C
        Label(fig[0, c], "Community $c"; tellwidth=false)
    end
    for l in 1:L
        Label(fig[l, C + 1], "Layer $(l-1)"; tellheight=false)
        axes = [
            Axis(
                fig[l, c];
                aspect=1,
                xticksvisible=l == L,
                xticklabelsvisible=l == L,
                yticksvisible=c == 1,
                yticklabelsvisible=c == 1,
            ) for c in 1:C
        ]
        append!(all_axes, axes)
        for ax in axes
            linkxaxes!(all_axes[1], ax)
            linkyaxes!(all_axes[1], ax)
        end

        lmin, lmax = extrema(mapreduce(vec, vcat, ls[l, :]))

        for c in 1:C
            scatter!(
                axes[c],
                histograms[l][c][:, 1],
                histograms[l][c][:, 2];
                color=colors[c],
                label="community $c",
                alpha=1 / 10,
            )
            contour!(
                axes[c],
                xrange,
                yrange,
                ls[l, c];
                colorrange=(lmin, lmax),
                colormap=:plasma,
                levels=15,
                linewidth=2,
            )
        end
        # Colorbar(
        #     fig[l, C + 1]; limits=(lmin, lmax), colormap=:plasma, label="Layer $l density"
        # )
    end

    resize_to_layout!(fig)
    return fig
end
