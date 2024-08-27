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
)
    (; sbm, features) = csbm
    L = size(densities, 1) - 1
    C = nb_communities(sbm)

    joint_histogram_scalar = mapreduce(vec, vcat, vec(embeddings))
    xrange = range(extrema(joint_histogram_scalar)..., 200)

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    with_theme(theme) do
        fig = Figure(; size=(600, 200 * (L + 1)))
        axes = Axis[]
        Label(fig[0, 1], "Contextual SBM in 1D"; tellwidth=false, fontsize=20)
        for l in 0:L
            Label(fig[l + 1, 2], "Layer $l"; tellheight=false, rotation=1.5π)
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
        fig
    end
end

function Oversmoothing.plot_2d(
    csbm::CSBM,
    embeddings::Matrix{<:Matrix},
    densities::Matrix{<:Mixture};
    theme=theme_latexfonts(),
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
    ls = [logdensityof.(Ref(densities[l + 1, c]), zs) for l in 0:L, c in 1:C]

    colors = distinguishable_colors(C, [RGB(1, 1, 1), RGB(0, 0, 0)]; dropseed=true)

    with_theme(theme) do
        fig = Figure(; size=(700, 200 * (L + 1)))
        all_axes = Axis[]
        Label(fig[-1, 1:C], "Contextual SBM in 2D"; tellwidth=false, fontsize=20)
        for c in 1:C
            Label(fig[0, c], "Community $c"; tellwidth=false)
        end
        for l in 0:L
            Label(
                fig[l + 1, C + 1], "Layer $l"; tellheight=false, rotation=1.5π
            )
            axes = [
                Axis(
                    fig[l + 1, c];
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

            lmin, lmax = extrema(mapreduce(vec, vcat, ls[l + 1, :]))

            for c in 1:C
                scatter!(
                    axes[c],
                    embeddings[l + 1, c][:, 1],
                    embeddings[l + 1, c][:, 2];
                    color=colors[c],
                    label="community $c",
                    alpha=2 / 10,
                )
                contour!(
                    axes[c],
                    xrange,
                    yrange,
                    ls[l + 1, c];
                    colorrange=(lmin, lmax),
                    colormap=:grays,
                    levels=15,
                    linewidth=2,
                )
            end
        end
        resize_to_layout!(fig)
        return fig
    end
end

end
