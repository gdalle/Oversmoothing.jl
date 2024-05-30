function Oversmoothing.plot_2d_embeddings(H_split::Vector{<:AbstractMatrix})
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
