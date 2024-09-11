module OversmoothingCairoMakieExt

using CairoMakie
using Colors
using DensityInterface
using StaticArrays
using Oversmoothing

function Oversmoothing.plot_1d(
    csbm::CSBM,
    embeddings::Matrix{<:Matrix},
    densities::Matrix{<:Mixture};
    theme=theme_latexfonts(),
    figsize=(500, 500),
    path=nothing,
)
    (; sbm, features) = csbm
    L = size(densities, 1) - 1
    C = nb_communities(sbm)

    joint_histogram_scalar = mapreduce(vec, vcat, vec(embeddings))
    xrange = range(extrema(joint_histogram_scalar)..., 200)

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    with_theme(theme) do
        fig = Figure(; size=figsize)
        axes = Axis[]
        for l in 0:L
            Label(fig[l + 1, 2], "layer $l"; tellheight=false, rotation=1.5π)
            ax = Axis(fig[l + 1, 1]; xticksvisible=l == L, xticklabelsvisible=l == L)
            push!(axes, ax)
            linkxaxes!(ax, axes[1])
            for c in 1:C
                hist!(
                    ax,
                    vec(embeddings[l + 1, c]);
                    normalization=:pdf,
                    bins=50,
                    label="community $c",
                    color=(colors[c], 0.5),
                )
                lines!(
                    ax,
                    xrange,
                    [densityof(densities[l + 1, c], [x]) for x in xrange];
                    color=colors[c],
                    linewidth=2,
                )
            end
        end
        Legend(fig[0, 1], first(axes); tellwidth=false, orientation=:horizontal)
        if !isnothing(path)
            save(path, fig)
        end
        fig
    end
end

function Oversmoothing.plot_2d(
    csbm::CSBM,
    embeddings::Matrix{<:Matrix},
    densities::Matrix{<:Mixture};
    theme=theme_latexfonts(),
    figsize=(500, 500),
    path=nothing,
)
    (; sbm, features) = csbm
    C = nb_communities(sbm)
    L = size(embeddings, 1) - 1

    joint_histogram = reduce(vcat, vec(embeddings))
    joint_histogram_x = joint_histogram[:, 1]
    joint_histogram_y = joint_histogram[:, 2]
    xrange = range(extrema(joint_histogram_x)..., 100)
    yrange = range(extrema(joint_histogram_y)..., 100)
    zs = SVector.(xrange, yrange')
    ls = [densityof.(Ref(densities[l + 1, c]), zs) for l in 0:L, c in 1:C]

    combined_densities = [
        Mixture(densities[l + 1, :], sbm.sizes ./ sum(sbm.sizes)) for l in 0:L
    ]
    combined_ls = [densityof.(Ref(combined_densities[l + 1]), zs) for l in 0:L]

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    with_theme(theme) do
        fig = Figure(; size=figsize)
        all_axes = Axis[]
        for c in 1:C
            Label(fig[0, c], "community $c"; tellwidth=false)
        end
        Label(fig[0, C + 1], "all communities"; tellwidth=false)
        for l in 0:L
            axes = [
                Axis(
                    fig[l + 1, c];
                    aspect=1,
                    xticksvisible=l == L,
                    xticklabelsvisible=l == L,
                    yticksvisible=c == 1,
                    yticklabelsvisible=c == 1,
                ) for c in 1:(C + 1)
            ]
            append!(all_axes, axes)
            for ax in axes
                linkxaxes!(all_axes[1], ax)
                linkyaxes!(all_axes[1], ax)
            end

            lmin, lmax = extrema(mapreduce(vec, vcat, ls[l + 1, :]))

            for c in 1:C
                scatter!(
                    axes[c],
                    embeddings[l + 1, c][:, 1],
                    embeddings[l + 1, c][:, 2];
                    color=colors[c],
                    label="community $c",
                    alpha=3 / 100,
                )
                contour!(
                    axes[c],
                    xrange,
                    yrange,
                    ls[l + 1, c];
                    colorrange=(lmin, lmax),
                    colormap=:grays,
                    levels=5,
                    linewidth=1,
                )
            end

            contour!(
                axes[C + 1],
                xrange,
                yrange,
                combined_ls[l + 1];
                colorrange=(lmin, lmax),
                colormap=:grays,
                levels=5,
                linewidth=1,
            )

            Label(fig[l + 1, C + 2], "layer $l"; tellheight=false, rotation=1.5π)
        end
        resize_to_layout!(fig)
        if !isnothing(path)
            save(path, fig)
        end
        return fig
    end
end

end
