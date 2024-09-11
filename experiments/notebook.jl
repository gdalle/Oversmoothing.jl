### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 10fa26e0-58a9-11ef-1536-e9fc7dc3721e
begin
    using Pkg
    Pkg.activate(@__DIR__)
	Pkg.instantiate()

    using CairoMakie
    using DensityInterface
    using LaTeXStrings
    using LinearAlgebra
    using Oversmoothing
    using PlutoUI
    using ProgressLogging
    using Random
    using StableRNGs

    BLAS.set_num_threads(1)
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

# ╔═╡ e546264c-6666-4138-8e96-bd6f33f54a07
IMG_PATH = joinpath(@__DIR__, "images")

# ╔═╡ 3c9cb7de-33eb-4422-ae50-95d5bf7484e0
md"""
## Toy CSBMs
"""

# ╔═╡ c5a94b08-78dc-4dd6-96c2-7ebe488205d7
let
	rng = StableRNG(63)
    C = 6
    csbm = LinearCSBM1d(; N=100, C=C, p_in=0.03, p_out=0.01, σ=0.1)

    fig = Figure(size=(450, 300))
    ax = Axis(fig[1, 1], xlabel="only feature", ylabel="density", xticks=1:C)
    for c in 1:C
        x = [rand(rng, csbm.features[c]) for _ in 1:500]
        hist!(ax, first.(x), normalization=:pdf, label="community $c")
    end
	Legend(fig[0, 1], ax, orientation=:horizontal, nbanks=2)
    fig
end

# ╔═╡ c1a99529-cf54-4ff7-ae44-4bdfce781a07
let
	rng = StableRNG(63)
    C = 6
    csbm = CircularCSBM2d(; N=100, C=C, p_in=0.03, p_out=0.01, σ=0.1, stretch=1)

    fig = Figure(size=(450, 300))
    ax = Axis(fig[1, 1]; aspect=1, xlabel="feature 1", ylabel="feature 2")
    for c in 1:C
        x = [rand(rng, csbm.features[c]) for _ in 1:100]
        scatter!(ax, first.(x), last.(x), label="community $c", alpha=0.5)
    end
	Legend(fig[1, 2], ax, orientation=:vertical)
	rowsize!(fig.layout, 1, Aspect(1, 1.0))
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
	rng = StableRNG(63)
    csbm = LinearCSBM1d(; N=100, C=3, p_in=0.03, p_out=0.02, σ=0.1)

    nb_layers = 2
    nb_graphs = 100
    embeddings = empirical_embeddings(rng, csbm; nb_layers, nb_graphs)
    densities = random_walk_densities(rng, csbm; nb_layers, nb_graphs)

    fig = plot_1d(csbm, embeddings, densities; theme=MYTHEME, figsize=(600, 400))
	save(joinpath(IMG_PATH, "illustration_1d.png"), fig; px_per_unit=5)
	fig
end

# ╔═╡ 67987f14-d637-4f33-b3e9-91597290cb74
md"""
## 2D
"""

# ╔═╡ 0735dfc4-85c0-491b-8bb6-58aa4272b772
let
	rng = StableRNG(63)
    csbm = CircularCSBM2d(; N=100, C=3, p_in=0.03, p_out=0.02, σ=0.1)

    nb_layers = 2
    nb_graphs = 100
    embeddings = empirical_embeddings(rng, csbm; nb_layers, nb_graphs)
    densities = random_walk_densities(rng, csbm; nb_layers, nb_graphs)

    fig = plot_2d(csbm, embeddings, densities; theme=MYTHEME, figsize=(700, 500))
	save(joinpath(IMG_PATH, "illustration_2d.png"), fig; px_per_unit=5)
	fig
end

# ╔═╡ f53e238d-6f08-4da0-a5af-7278a7c64e5c
md"""
# First layer
"""

# ╔═╡ c69e1c1f-e5c2-41cf-86c3-6a25c6ff5a9f
md"""
## Normality
"""

# ╔═╡ cc21e14f-12b1-4cb9-972e-f90750b7a551
@kwdef struct NormalityExperiment
	N_vals
	C
	p_in
	p_out
	σ_vals
	total_variation_vals
end

# ╔═╡ 04613bc4-c04a-4497-a024-8c8baf13f755
normality_experiment = let
	rng = StableRNG(63)

	N_vals = 50:10:500
	C = 2
	p_in = 0.03
	p_out = 0.02
	σ_vals = 0.01:0.005:0.2

	total_variation_vals = fill(NaN, length(N_vals), length(σ_vals))
	
	@progress for i in eachindex(N_vals), j in eachindex(σ_vals)
		N, σ = N_vals[i], σ_vals[j]
		csbm = LinearCSBM1d(; N, C, p_in, p_out, σ)

		density1 = first_layer_densities(csbm)[1]
		density1_normal = MultivariateNormal(density1)
		tv = total_variation_quadrature(density1, density1_normal; rtol=1e-5)
		total_variation_vals[i, j] = value(tv)
	end

	NormalityExperiment(; N_vals, C, p_in, p_out, σ_vals, total_variation_vals)
end

# ╔═╡ 81ba0e8c-49e0-4744-b9e8-fffa2b05aaaa
let
	(; N_vals, σ_vals, total_variation_vals) = normality_experiment

	fig = Figure(size=(500, 400))
    ax = Axis(fig[1, 1]; xlabel=L"graph size $N$", ylabel=L"noise $\sigma$")
    hm = heatmap!(
        ax,
        N_vals,
        σ_vals,
        total_variation_vals;
        colormap=:viridis,
		colorscale=log10,
    )
    Colorbar(fig[1, 2], hm; label="total variation distance \nbetween mixture and Gaussian")
	rowsize!(fig.layout, 1, Aspect(1, 1))
	save(joinpath(IMG_PATH, "distance_mixture_gaussian.png"), fig; px_per_unit=5)
    fig
end

# ╔═╡ 4b8d0758-7c36-42d4-b2c7-df5a4b033d38
md"""
## Connectivity
"""

# ╔═╡ 4b7d7234-ece3-4875-a232-562ebe418055
@kwdef struct ConnectivityExperiment
	N
	C
	p_in_vals
	p_out_vals
	σ
	accuracy0_vals
	accuracy1_vals
end

# ╔═╡ c5ddce4c-21ae-4eb9-a411-a576ac8f766d
connectivity_experiment = let
	rng = StableRNG(63)
	N = 100
    C = 2
    σ = 1.0

    p_in_vals = (0:0.2:10) ./ N
    p_out_vals = (0:0.2:10) ./ N

	accuracy0_vals = fill(NaN, length(p_in_vals), length(p_out_vals))
    accuracy1_vals = fill(NaN, length(p_in_vals), length(p_out_vals))

    @progress for i in eachindex(p_in_vals), j in eachindex(p_out_vals)
        p_in, p_out = p_in_vals[i], p_out_vals[j]
        csbm = LinearCSBM1d(; N, C, p_in, p_out, σ)
        accuracy0_vals[i, j] = accuracy_zeroth_layer(csbm; rtol=1e-4) |> value
        accuracy1_vals[i, j] = accuracy_first_layer(csbm; rtol=1e-4) |> value
    end
    
	ConnectivityExperiment(;
		N, C, p_in_vals, p_out_vals, σ, accuracy0_vals, accuracy1_vals
	)
end

# ╔═╡ 2b272855-8b04-47d0-b2c5-c672ab633f79
let
	(; p_in_vals, p_out_vals, accuracy0_vals, accuracy1_vals) = connectivity_experiment

	acc_diff_vals = accuracy1_vals .- accuracy0_vals

    fig = Figure(size=(500, 450))
    ax = Axis(fig[1, 1]; aspect=1, xlabel=L"inner connectivity $p_{in}$", ylabel=L"outer connectivity $p_{out}$")
    hm = heatmap!(
        ax,
        p_in_vals,
        p_out_vals,
        acc_diff_vals;
        colormap=Reverse(:curl),
        colorrange=(-maximum(abs, acc_diff_vals), maximum(abs, acc_diff_vals)),
    )
    Colorbar(fig[1, 2], hm; label=L"accuracy improvement after $1$ layer")
	rowsize!(fig.layout, 1, Aspect(1, 1))
	save(joinpath(IMG_PATH, "onelayer_improvement_p.png"), fig; px_per_unit=5)
    fig
end

# ╔═╡ 9a881c91-fffb-416a-beb9-ad5985063273
md"""
# More layers
"""

# ╔═╡ 4c090dd7-f296-4836-bee4-0ed43a3ea7ec
md"""
## Oscillations
"""

# ╔═╡ 539f2393-3ba8-4287-8e82-aa602e76cf01
@kwdef struct OscillationsExperiment
	N
	C
	p_in
	p_out
	σ
	accuracies_rw
	accuracies_lr
end

# ╔═╡ 9ba113e6-b1bf-4f05-a185-2b6e77f453ab
oscillations_experiments = map(2:5) do C
	@info "C = $C"
	rng = StableRNG(63)
	N = 100
	p_in = 0.05
	p_out = 0.01
	σ = 0.5
	
	nb_layers, nb_trajectories, nb_graphs = 5, 20, 50
    
	csbm = CircularCSBM2d(; N, C, p_in, p_out, σ)

    accuracies_rw = accuracy_by_depth(
		rng, csbm; method=:randomwalk,
		nb_layers, nb_trajectories, nb_graphs
	)
	accuracies_lr = accuracy_by_depth(
		rng, csbm; method=:logisticregression,
		nb_layers, nb_trajectories, nb_graphs
	)

    OscillationsExperiment(; N, C, p_in, p_out, σ, accuracies_rw, accuracies_lr)
end

# ╔═╡ a52d2a3a-4e6a-40a9-90ab-543d3e021f60
let
	fig = Figure()
	ax = Axis(fig[1:2, 1]; xlabel=L"depth $L$", ylabel=L"accuracy $a$")

	bands = []
	scatters = []
	errors = []

	C_vals = []
	
	for (j, oscillations_experiment) in enumerate(oscillations_experiments)
		(; C, accuracies_rw, accuracies_lr) = oscillations_experiment
		nb_layers = length(accuracies_rw) - 1
	
		val_rw, unc_rw = value.(accuracies_rw), uncertainty.(accuracies_rw)
		val_lr, unc_lr = value.(accuracies_lr), uncertainty.(accuracies_lr)

		b = band!(ax, 0:nb_layers, val_lr - unc_lr, val_lr + unc_lr, label=L"regression - $C=%$C$", alpha=0.3)
	    s = scatterlines!(ax, 0:nb_layers, val_rw, label=L"theory - $C=%$C$", markersize=0)
	    e = errorbars!(ax, 0:nb_layers, val_rw, unc_rw, linewidth=2)
		
		push!(bands, b)
		push!(scatters, s)
		push!(errors, e)
		push!(C_vals, C)

		local_extrema = [
			L for L in 1:(nb_layers-1) if (
				(val_rw[L+1] > max(val_rw[L], val_rw[L+2])) ||
				(val_rw[L+1] < min(val_rw[L], val_rw[L+2]))
			)
		]
		scatter!(ax, local_extrema, val_rw[local_extrema .+ 1], markersize=45, marker=:circle, alpha=0, strokecolor=:black, strokewidth=3)
	end

	for c in C_vals
		
	end

	C_strings = [L"C = %$C" for C in C_vals]
	Legend(fig[1, 2], scatters, C_strings, "Mixture")
	Legend(fig[2, 2], bands, C_strings, "Regression")
	save(joinpath(IMG_PATH, "optimal_depth.png"), fig; px_per_unit=5)
    fig
end

# ╔═╡ Cell order:
# ╟─d46ca486-6b19-4b00-b996-2762d683eb1e
# ╟─9733cd58-fbee-4c0e-839d-15fc362c9abf
# ╠═10fa26e0-58a9-11ef-1536-e9fc7dc3721e
# ╠═652e47fb-56d1-4afe-afea-6f551ec39346
# ╠═8b80f3b5-4fdd-48c8-9f0d-52e35535a653
# ╠═e546264c-6666-4138-8e96-bd6f33f54a07
# ╟─3c9cb7de-33eb-4422-ae50-95d5bf7484e0
# ╠═c5a94b08-78dc-4dd6-96c2-7ebe488205d7
# ╠═c1a99529-cf54-4ff7-ae44-4bdfce781a07
# ╟─38ddf907-2f83-4a43-89bb-f65356b6445b
# ╟─3fadb088-ef57-47c0-b564-8a2d268b514a
# ╠═a7e26ada-e581-4ef6-91ca-f0dba906ebb8
# ╟─67987f14-d637-4f33-b3e9-91597290cb74
# ╠═0735dfc4-85c0-491b-8bb6-58aa4272b772
# ╟─f53e238d-6f08-4da0-a5af-7278a7c64e5c
# ╟─c69e1c1f-e5c2-41cf-86c3-6a25c6ff5a9f
# ╠═cc21e14f-12b1-4cb9-972e-f90750b7a551
# ╠═04613bc4-c04a-4497-a024-8c8baf13f755
# ╠═81ba0e8c-49e0-4744-b9e8-fffa2b05aaaa
# ╟─4b8d0758-7c36-42d4-b2c7-df5a4b033d38
# ╠═4b7d7234-ece3-4875-a232-562ebe418055
# ╠═c5ddce4c-21ae-4eb9-a411-a576ac8f766d
# ╠═2b272855-8b04-47d0-b2c5-c672ab633f79
# ╟─9a881c91-fffb-416a-beb9-ad5985063273
# ╟─4c090dd7-f296-4836-bee4-0ed43a3ea7ec
# ╠═539f2393-3ba8-4287-8e82-aa602e76cf01
# ╠═9ba113e6-b1bf-4f05-a185-2b6e77f453ab
# ╠═a52d2a3a-4e6a-40a9-90ab-543d3e021f60
