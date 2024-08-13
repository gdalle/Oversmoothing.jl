### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 10fa26e0-58a9-11ef-1536-e9fc7dc3721e
begin
    using Revise
    using Pkg
    Pkg.activate(@__DIR__)

    using CairoMakie
    using DensityInterface
    using LaTeXStrings
    using LinearAlgebra
    using MonteCarloMeasurements
    using OhMyThreads
    using Oversmoothing
    using PlutoUI
    using ProgressLogging
    using Random
    using StableRNGs
    using StaticArrays

    BLAS.set_num_threads(1)
    rng = Random.default_rng()
end

# ╔═╡ d46ca486-6b19-4b00-b996-2762d683eb1e
md"""
# Preliminaries
"""

# ╔═╡ 9733cd58-fbee-4c0e-839d-15fc362c9abf
md"""
## Imports
"""

# ╔═╡ 652e47fb-56d1-4afe-afea-6f551ec39346
TableOfContents()

# ╔═╡ 963d8b7d-6b08-4f61-9133-d848353fec46
md"""
## Settings
"""

# ╔═╡ 8b80f3b5-4fdd-48c8-9f0d-52e35535a653
begin
    MYTHEME = merge(
        theme_latexfonts(),
        Theme(;
            palette=(
                color=Makie.wong_colors(),
                linestyle=[:solid, :dash, :dashdot, :dot],
                marker=[:circle, :xcross, :rect, :star5, :utriangle],
            ),
            Scatter=(cycle=Cycle([:color, :linestyle, :marker]; covary=true),),
            ScatterLines=(cycle=Cycle([:color, :linestyle, :marker]; covary=true),),
        ),
    )

    Makie.set_theme!(MYTHEME)
end

# ╔═╡ efa56462-c94c-4b21-96e7-88ac5cfa9be1
begin
    nb_layers = 10
    nb_trajectories = 10
    nb_graphs = 100
    nb_samples = 200
end

# ╔═╡ 3c9cb7de-33eb-4422-ae50-95d5bf7484e0
md"""
## Toy CSBMs
"""

# ╔═╡ 1ea22ce3-0423-4a80-9fbe-2b1d396f7f64
function LinearCSBM1d(; C::Integer, din::Real, dout::Real, σ::Real, N::Integer=100)
    p = din / N
    q = dout / N
    sbm = SBM(N, C, p, q)
    μ = float.(1:C)
    Σ = fill(σ^2, C)
    features = UnivariateNormal.(μ, Σ)
    return CSBM(sbm, features)
end

# ╔═╡ c5a94b08-78dc-4dd6-96c2-7ebe488205d7
let
    C = 4
    csbm = LinearCSBM1d(; C=C, din=3, dout=2, σ=0.1, N=100)

    fig = Figure()
    ax = Axis(fig[1, 1])
    for c in 1:C
        x = [rand(rng, csbm.features[c]) for _ in 1:100]
        hist!(ax, first.(x))
    end
    fig
end

# ╔═╡ 09ef5298-50fa-40f5-978d-24c8db4ff6e9
function SymmetricCSBM2d(; C::Integer, din::Real, dout::Real, σ::Real, N::Integer=100)
    p = din / N
    q = dout / N
    sbm = SBM(N, C, p, q)
    μ = [[cospi(2(c - 1) / C), sinpi(2(c - 1) / C)] for c in 1:C]
    Σ = [Diagonal([σ^2, σ^2]) for c in 1:C]
    features = BivariateNormal.(μ, Σ)
    return CSBM(sbm, features)
end

# ╔═╡ c1a99529-cf54-4ff7-ae44-4bdfce781a07
let
    C = 10
    csbm = SymmetricCSBM2d(; C=C, din=3, dout=2, σ=0.1, N=100)

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=1)
    for c in 1:C
        x = [rand(rng, csbm.features[c]) for _ in 1:100]
        scatter!(ax, first.(x), last.(x))
    end
    fig
end

# ╔═╡ 38ddf907-2f83-4a43-89bb-f65356b6445b
md"""
# Illustrations
"""

# ╔═╡ 3fadb088-ef57-47c0-b564-8a2d268b514a
md"""
## 1D
"""

# ╔═╡ a7e26ada-e581-4ef6-91ca-f0dba906ebb8
let
    sbm = SBM(300, 3, 0.03, 0.01)
    features = [
        UnivariateNormal(-2.0, 0.03),
        UnivariateNormal(-0.0, 0.01),
        UnivariateNormal(+1.0, 0.02),
    ]
    csbm = CSBM(sbm, features)

    nb_layers = 2
    nb_graphs = 100
    histograms = embeddings(rng, csbm; nb_layers, nb_graphs)
    densities = random_walk_densities(rng, csbm; nb_layers, nb_graphs)

    plot_1d(csbm, histograms, densities; theme=MYTHEME)
end

# ╔═╡ 67987f14-d637-4f33-b3e9-91597290cb74
md"""
## 2D
"""

# ╔═╡ 0735dfc4-85c0-491b-8bb6-58aa4272b772
let
    sbm = SBM(300, 3, 0.03, 0.01)
    features = [
        BivariateNormal([-2.0, 0.0], [1.0 0.2; 0.2 2.0]),
        BivariateNormal([0.0, 2.0], [2.0 -0.4; -0.4 1.0]),
        BivariateNormal([+3.0, 0.0], [1.0 0.3; 0.3 1.0]),
    ]
    csbm = CSBM(sbm, features)

    nb_layers = 2
    nb_graphs = 100
    histograms = embeddings(rng, csbm; nb_layers, nb_graphs)
    densities = random_walk_densities(rng, csbm; nb_layers, nb_graphs)

    plot_2d(csbm, histograms, densities; theme=MYTHEME)
end

# ╔═╡ f53e238d-6f08-4da0-a5af-7278a7c64e5c
md"""
# First layer
"""

# ╔═╡ 4b8d0758-7c36-42d4-b2c7-df5a4b033d38
md"""
## Connectivities
"""

# ╔═╡ c5ddce4c-21ae-4eb9-a411-a576ac8f766d
din_dout_error0, din_dout_error1_vals = let
    C = 2
    σ = 2.0

    din_vals = 0:0.2:10
    dout_vals = 0:0.2:10

    error0 = error_zeroth_layer(LinearCSBM1d(; C, din=0, dout=0, σ); rtol=1e-4)
    error1_vals = fill(NaN, length(din_vals), length(dout_vals))

    @progress for i in eachindex(din_vals), j in eachindex(dout_vals)
        din, dout = din_vals[i], dout_vals[j]
        csbm = LinearCSBM1d(; C, din, dout, σ)
        error1_vals[i, j] = error_first_layer(
            csbm; max_neighbors=ceil(Int, 30(din + dout)), rtol=1e-4
        )
    end
    error0, error1_vals
end

# ╔═╡ 2b272855-8b04-47d0-b2c5-c672ab633f79
let
    din_vals = 0:0.2:10
    dout_vals = 0:0.2:10
    err_diff_vals = din_dout_error1_vals .- din_dout_error0

    fig = Figure()
    ax = Axis(fig[1, 1]; aspect=1, xlabel=L"d_{in}", ylabel=L"d_{out}")
    hm = heatmap!(
        ax,
        din_vals,
        dout_vals,
        err_diff_vals;
        colormap=:balance,
        colorrange=(-maximum(abs, err_diff_vals), maximum(abs, err_diff_vals)),
    )
    Colorbar(fig[1, 2], hm; label=L"error difference after $1$ layer")
    fig
end

# ╔═╡ c38a15fc-ba48-4c01-b433-c18671435598
md"""
## Variance
"""

# ╔═╡ 56833a00-b483-4d82-8eae-1825783668d2
C_σ_error0_vals, C_σ_error1_vals = let
    C_vals = 2:4
    σ_vals = 0.05:0.05:3.0

    error0_vals = fill(NaN, length(C_vals), length(σ_vals))
    error1_vals = fill(NaN, length(C_vals), length(σ_vals))

    @progress for i in eachindex(C_vals), j in eachindex(σ_vals)
        C = C_vals[i]
        σ = σ_vals[j]

        csbm = LinearCSBM1d(; C, din=4, dout=1, σ, N=60)

        error0_vals[i, j] = error_zeroth_layer(csbm; rtol=1e-4)
        error1_vals[i, j] = error_first_layer(csbm; max_neighbors=20, rtol=1e-4)
    end

    error0_vals, error1_vals
end

# ╔═╡ ad02d145-9c0e-447f-adb1-535d3f1d46cd
let
    C_vals = 2:4
    σ_vals = 0.05:0.05:3.0

    err_diff_vals = C_σ_error1_vals - C_σ_error0_vals

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"\sigma", ylabel="error difference after 1 layer")
    hlines!(ax, [0.0]; color=:black, linewidth=3)
    for (k, C) in enumerate(C_vals)
        scatterlines!(σ_vals, err_diff_vals[k, :]; label=L"C=%$C")
    end
    Legend(fig[1, 2], ax)
    fig
end

# ╔═╡ 9a881c91-fffb-416a-beb9-ad5985063273
md"""
# More layers
"""

# ╔═╡ 3a161f37-16ec-4604-b959-7104c4535b7e
md"""
## Variance
"""

# ╔═╡ 38202eb9-7103-412a-9ccd-3d8ca69bb71e
let
    C = 2
    din = 5
    dout = 1
    σ_vals = 0.2:0.4:2

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="depth", ylabel="error")
    @progress for (k, σ) in enumerate(σ_vals)
        csbm = SymmetricCSBM2d(; C, din, dout, σ)
        errors = error_by_depth(
            rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples
        )
        scatterlines!(ax, 0:nb_layers, pmean.(errors); label="$σ")
        errorbars!(ax, 0:nb_layers, pmean.(errors), pstd.(errors))
    end
    Legend(fig[1, 2], ax, L"\sigma")
    fig
end

# ╔═╡ 7c6d090b-5d49-4a7e-adc3-be9028a74097
md"""
## Connectivity
"""

# ╔═╡ 220429bd-54d3-4c63-9540-cc648d8e9167
let
    C = 2
    din_vals = 2:6
    dout = 1
    σ = 1.0

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="depth", ylabel="error")
    @progress for (k, din) in enumerate(din_vals)
        csbm = SymmetricCSBM2d(; C, din, dout, σ)
        errors = error_by_depth(
            rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples
        )
        scatterlines!(ax, 0:nb_layers, pmean.(errors); label="$din")
        errorbars!(ax, 0:nb_layers, pmean.(errors), pstd.(errors))
    end
    Legend(fig[1, 2], ax, L"d_{in}")
    fig
end

# ╔═╡ 4c090dd7-f296-4836-bee4-0ed43a3ea7ec
md"""
## Oscillations
"""

# ╔═╡ 9ba113e6-b1bf-4f05-a185-2b6e77f453ab
let
    csbm = CSBM(
        SBM(100, 2, 0.05, 0.05), [UnivariateNormal(-1.0, 1.0), UnivariateNormal(+1.0, 1.0)]
    )

    nb_layers = 10

    errors = error_by_depth(rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="depth", ylabel="error")
    scatterlines!(ax, 0:nb_layers, pmean.(errors))
    errorbars!(ax, 0:nb_layers, pmean.(errors), pstd.(errors))
    fig
end

# ╔═╡ Cell order:
# ╟─d46ca486-6b19-4b00-b996-2762d683eb1e
# ╟─9733cd58-fbee-4c0e-839d-15fc362c9abf
# ╠═10fa26e0-58a9-11ef-1536-e9fc7dc3721e
# ╠═652e47fb-56d1-4afe-afea-6f551ec39346
# ╟─963d8b7d-6b08-4f61-9133-d848353fec46
# ╠═8b80f3b5-4fdd-48c8-9f0d-52e35535a653
# ╠═efa56462-c94c-4b21-96e7-88ac5cfa9be1
# ╟─3c9cb7de-33eb-4422-ae50-95d5bf7484e0
# ╠═1ea22ce3-0423-4a80-9fbe-2b1d396f7f64
# ╠═c5a94b08-78dc-4dd6-96c2-7ebe488205d7
# ╠═09ef5298-50fa-40f5-978d-24c8db4ff6e9
# ╠═c1a99529-cf54-4ff7-ae44-4bdfce781a07
# ╟─38ddf907-2f83-4a43-89bb-f65356b6445b
# ╟─3fadb088-ef57-47c0-b564-8a2d268b514a
# ╠═a7e26ada-e581-4ef6-91ca-f0dba906ebb8
# ╟─67987f14-d637-4f33-b3e9-91597290cb74
# ╠═0735dfc4-85c0-491b-8bb6-58aa4272b772
# ╟─f53e238d-6f08-4da0-a5af-7278a7c64e5c
# ╟─4b8d0758-7c36-42d4-b2c7-df5a4b033d38
# ╠═c5ddce4c-21ae-4eb9-a411-a576ac8f766d
# ╠═2b272855-8b04-47d0-b2c5-c672ab633f79
# ╟─c38a15fc-ba48-4c01-b433-c18671435598
# ╠═56833a00-b483-4d82-8eae-1825783668d2
# ╠═ad02d145-9c0e-447f-adb1-535d3f1d46cd
# ╟─9a881c91-fffb-416a-beb9-ad5985063273
# ╟─3a161f37-16ec-4604-b959-7104c4535b7e
# ╠═38202eb9-7103-412a-9ccd-3d8ca69bb71e
# ╟─7c6d090b-5d49-4a7e-adc3-be9028a74097
# ╠═220429bd-54d3-4c63-9540-cc648d8e9167
# ╟─4c090dd7-f296-4836-bee4-0ed43a3ea7ec
# ╠═9ba113e6-b1bf-4f05-a185-2b6e77f453ab
