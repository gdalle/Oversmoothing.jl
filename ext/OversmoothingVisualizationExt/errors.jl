function misclassification_probability_evolution(
    rng::AbstractRNG,
    graph::AbstractRandomGraph,
    features::Vector;
    max_layers::Integer,
    resample_graph::Bool,
)
    _, history_train = embeddings(
        rng, graph, features; layers=max_layers, resample_graph, return_history=true
    )
    _, history_test = embeddings(
        rng, graph, features; layers=max_layers, resample_graph, return_history=true
    )

    probas = Vector{Float64}(undef, length(history_train))
    tforeach(eachindex(history_train, history_test)) do l
        H_split_train = split_by_community(history_train[l], graph)
        H_split_test = split_by_community(history_test[l], graph)
        probas[l] = misclassification_probability(H_split_train, H_split_test)
    end

    return probas
end

function Oversmoothing.plot_misclassification(
    rng::AbstractRNG,
    graph::AbstractRandomGraph,
    features::Vector;
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
    errorbars!(ax, (0:max_layers) .- 0.1, probas_no_resample_mean, probas_no_resample_std;)

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
