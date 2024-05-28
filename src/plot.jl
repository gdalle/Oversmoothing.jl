function plot_1d_embeddings(H_split::Vector{<:AbstractMatrix})
    linestyles = [:dash, :dashdot, :dashdotdot]
    C = length(H_split)
    H = reduce(vcat, H_split)
    H_range = range(minimum(H[:, 1]), maximum(H[:, 1]); length=100)

    fig = Figure()

    ax0 = Axis(fig[1, 1]; title="Embedding distribution of a GCN on the CSBM")
    h0 = hist!(
        ax0, H[:, 1]; normalization=:pdf, color=(:black, 0.5), label="all communities"
    )
    P = density_estimator(H)
    l0 = lines!(ax0, H_range, pdf.(Ref(P), H_range); color=:black, linewidth=2)

    Legend(fig[1, 2], [[h0, l0]], ["all communities"])

    ax = Axis(fig[2, 1])
    linkxaxes!(ax, ax0)

    legend_objects = []
    legend_names = []

    for c in 1:C
        hc = hist!(
            ax,
            H_split[c][:, 1];
            normalization=:pdf,
            color=c,
            colormap=(:tab10, 0.5),
            colorrange=(1, 10),
        )
        push!(legend_objects, hc)
        push!(legend_names, "community $c (size $(length(H_split[c])))")
    end

    for c in 1:C
        Pc = density_estimator(H_split[c])
        lc = lines!(
            ax,
            H_range,
            pdf.(Ref(Pc), H_range);
            color=c,
            colormap=:tab10,
            colorrange=(1, 10),
            linewidth=2,
            linestyle=linestyles[c],
        )

        legend_objects[c] = [legend_objects[c], lc]
    end

    Legend(fig[2, 2], legend_objects, legend_names)
    return fig
end

function plot_2d_embeddings(H_split::Vector{<:AbstractMatrix})
    C = length(H_split)
    H = reduce(vcat, H_split)
    H_xrange = range(minimum(H[:, 1]), maximum(H[:, 1]); length=30)
    H_yrange = range(minimum(H[:, 2]), maximum(H[:, 2]); length=30)

    fig = Figure()

    supertitle = Label(
        fig[0, 1:2], "Embedding distribution of a GCN on the CSBM"; fontsize=22
    )

    ax0_bins = Axis(fig[1, 1]; aspect=1, title="Empirical distribution")
    hb0 = hexbin!(ax0_bins, H[:, 1], H[:, 2])

    ax0_dens = Axis(fig[1, 2]; aspect=1, title="Density estimate")
    linkxaxes!(ax0_bins, ax0_dens)
    linkyaxes!(ax0_bins, ax0_dens)
    P = density_estimator(H)
    z = pdf.(Ref(P), H_xrange, transpose(H_yrange))
    cont0 = contour!(ax0_dens, H_xrange, H_yrange, z)
    #=
    for c in 1:C
        axc_bins = Axis(fig[1 + c, 1]; aspect=1)
        linkxaxes!(ax0_bins, axc_bins)
        linkyaxes!(ax0_bins, axc_bins)
        hbc = hexbin!(axc_bins, H_split[c][:, 1], H_split[c][:, 2])

        axc_dens = Axis(fig[1 + c, 2]; aspect=1)
        linkxaxes!(ax0_bins, axc_dens)
        linkyaxes!(ax0_bins, axc_dens)
        Pc = density_estimator(H_split[c])
        zc = pdf.(Ref(Pc), H_xrange, transpose(H_yrange))
        contc = contour!(axc_dens, H_xrange, H_yrange, zc)
    end
    =#

    resize_to_layout!(fig)
    return fig
end

function plot_misclassification(
    rng::AbstractRNG,
    graph::AbstractRandomGraph,
    features::Vector{<:MultivariateDistribution};
    max_layers::Integer,
    graph_samples::Integer,
)
    prog = Progress(2 * graph_samples; desc="Sampling:")

    probas_no_resample = Vector{Vector{Float64}}(undef, graph_samples)
    probas_resample = Vector{Vector{Float64}}(undef, graph_samples)

    @threads for s in 1:graph_samples
        probas_no_resample[s] = misclassification_probability_evolution(
            rng, graph, features; max_layers, resample_graph=false
        )
        next!(prog)
        probas_resample[s] = misclassification_probability_evolution(
            rng, graph, features; max_layers, resample_graph=false
        )
        next!(prog)
    end

    probas_no_resample_mean = mean(probas_no_resample)
    probas_no_resample_std = std(probas_no_resample)

    probas_resample_mean = mean(probas_resample)
    probas_resample_std = std(probas_resample)

    fig = Figure()

    ax = Axis(
        fig[1, 1];
        xlabel="layers",
        ylabel="misclassification probability",
        title="$graph",
        xticks=0:max_layers,
    )

    scatterlines!(
        ax,
        (0:max_layers) .- 0.1,
        probas_no_resample_mean;
        label="no resampling",
        linestyle=:dash,
        marker=:diamond,
        markersize=15,
    )
    errorbars!(
        ax,
        (0:max_layers) .- 0.1,
        probas_no_resample_mean,
        probas_no_resample_std;
    )

    scatterlines!(
        ax,
        (0:max_layers) .+ 0.1,
        probas_resample_mean;
        label="resampling",
        linestyle=nothing,
        marker=:circle,
        markersize=15,
    )
    errorbars!(ax, (0:max_layers) .+ 0.1, probas_resample_mean, probas_resample_std;)

    axislegend(ax; position=:lt)
    return fig
end
