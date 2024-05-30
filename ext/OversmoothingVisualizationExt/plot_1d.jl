function Oversmoothing.plot_1d_embeddings(H_split::Vector{<:AbstractMatrix})
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
        hc = hist!(ax, H_split[c][:, 1]; normalization=:pdf)
        push!(legend_objects, hc)
        push!(legend_names, "community $c (size $(length(H_split[c])))")
    end

    for c in 1:C
        Pc = density_estimator(H_split[c])
        lc = lines!(
            ax, H_range, pdf.(Ref(Pc), H_range); linewidth=2, linestyle=linestyles[c]
        )

        legend_objects[c] = [legend_objects[c], lc]
    end

    Legend(fig[2, 2], legend_objects, legend_names)
    return fig
end
