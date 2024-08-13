### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 10fa26e0-58a9-11ef-1536-e9fc7dc3721e
begin
	using Revise
	using Pkg
	Pkg.activate(dirname(@__DIR__))

	using CairoMakie
	using DensityInterface
	using LaTeXStrings
	using Latexify
	using LinearAlgebra
	using MonteCarloMeasurements
	using OhMyThreads
	using Oversmoothing
	using PlutoUI
	using Random
	using StableRNGs
	using StaticArrays

	BLAS.set_num_threads(1)
	rng = Random.default_rng()
end

# ╔═╡ 9733cd58-fbee-4c0e-839d-15fc362c9abf
md"""
## Imports
"""

# ╔═╡ 652e47fb-56d1-4afe-afea-6f551ec39346
TableOfContents()

# ╔═╡ 7254e680-5fb8-424b-ac0c-db98e3724526
Makie.set_theme!(Oversmoothing.MYTHEME)

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
	histograms = embeddings(rng, csbm; nb_layers, nb_graphs);
	densities = random_walk_mixtures(rng, csbm; nb_layers, nb_graphs);
	
	plot_1d(csbm, histograms, densities)
end

# ╔═╡ 67987f14-d637-4f33-b3e9-91597290cb74
md"""
## 2D
"""

# ╔═╡ 0735dfc4-85c0-491b-8bb6-58aa4272b772
let
	sbm = SBM(300, 3, 0.03, 0.01)
	features = [
	    MultivariateNormal(SVector(-2.0, 0.0), SMatrix{2, 2}(1.0, 0.2, 0.2, 2.0)),
	    MultivariateNormal(SVector(0.0, 2.0), SMatrix{2, 2}(2.0, -0.4, -0.4, 1.0)),
	    MultivariateNormal(SVector(+3.0, 0.0), SMatrix{2, 2}(1.0, 0.3, 0.3, 1.0)),
	]
	csbm = CSBM(sbm, features)

	nb_layers = 2
	nb_graphs = 100
	histograms = embeddings(rng, csbm; nb_layers, nb_graphs);
	densities = random_walk_mixtures(rng, csbm; nb_layers, nb_graphs);
	
	plot_2d(csbm, histograms, densities)
end

# ╔═╡ 4d6d9739-baf0-4e65-8c86-4edaa50ceda0
A = rand(2, 2)

# ╔═╡ 5839e3f7-baf4-4b75-b401-373d1cf83e73


# ╔═╡ 3c9cb7de-33eb-4422-ae50-95d5bf7484e0
md"""
# Symmetric 2D CSBM
"""

# ╔═╡ 09ef5298-50fa-40f5-978d-24c8db4ff6e9
function SymmetricCSBM2d(;
    C::Integer,
    d_in::Real,
    d_out::Real,
    σ::Real,
	N::Integer = 100
)
	p = d_in / N
	q = d_out / N
    sbm = SBM(N, C, p, q)
    μ = [SVector(cospi(2(c - 1) / C), sinpi(2(c - 1) / C)) for c in 1:C]
    Σ = [SMatrix{2,2}(σ^2, 0.0, 0.0, σ^2) for c in 1:C]
    features = MultivariateNormal.(μ, Σ)
    return CSBM(sbm, features)
end

# ╔═╡ efa56462-c94c-4b21-96e7-88ac5cfa9be1
begin
	nb_layers = 5
	nb_trajectories = 10
	nb_graphs = 50
	nb_samples = 100
end

# ╔═╡ 3a161f37-16ec-4604-b959-7104c4535b7e
md"""
## Variance
"""

# ╔═╡ 38202eb9-7103-412a-9ccd-3d8ca69bb71e
let
	C = 2
	d_in = 5
	d_out = 1
	σ_vals = 0.2:0.4:2

	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="depth", ylabel="error")
	for (k, σ) in enumerate(σ_vals)
		csbm = SymmetricCSBM2d(; C, d_in, d_out, σ)
		errors = random_walk_errors(rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples)
		scatterlines!(ax, 0:nb_layers, pmean.(errors), label="$σ")
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
	d_in_vals = 2:6
	d_out = 1
	σ = 1.0

	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="depth", ylabel="error")
	for (k, d_in) in enumerate(d_in_vals)
		csbm = SymmetricCSBM2d(; C, d_in, d_out, σ)
		errors = random_walk_errors(rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples)
		scatterlines!(ax, 0:nb_layers, pmean.(errors), label="$d_in")
		errorbars!(ax, 0:nb_layers, pmean.(errors), pstd.(errors))
	end
	Legend(fig[1, 2], ax, L"d_{in}")
	fig
end

# ╔═╡ ec86695e-aea0-4176-9de3-03b859deec39
md"""
## Number of communities
"""

# ╔═╡ 60bde353-26d3-408f-8e78-31e6ec26cd6a
let
	C_vals = 2:2:10
	d_in = 5
	d_out = 1
	σ = 1.0
	
	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="depth", ylabel="error")
	for (k, C) in enumerate(C_vals)
		csbm = SymmetricCSBM2d(; C, d_in, d_out, σ)
		errors = random_walk_errors(rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples)
		scatterlines!(ax, 0:nb_layers, pmean.(errors), label="$C")
		errorbars!(ax, 0:nb_layers, pmean.(errors), pstd.(errors))
	end
	Legend(fig[1, 2], ax, L"C")
	fig
end

# ╔═╡ 4c090dd7-f296-4836-bee4-0ed43a3ea7ec
md"""
# Oscillations
"""

# ╔═╡ 9ba113e6-b1bf-4f05-a185-2b6e77f453ab
let
	C = 2
	csbm = CSBM(
		SBM(100, C, 0.05, 0.05),
		[UnivariateNormal(float(c), 1.0) for c in 1:C]
	)

	nb_layers = 10
	
	errors = random_walk_errors(rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples)

	fig = Figure()
	ax = Axis(fig[1, 1], xlabel="depth", ylabel="error")
	scatterlines!(ax, 0:nb_layers, pmean.(errors))
	errorbars!(ax, 0:nb_layers, pmean.(errors), pstd.(errors))
	fig
end

# ╔═╡ 58887711-76ea-4c2e-b55d-6b00f8c1f93a
md"""
## Two-timescale
"""

# ╔═╡ 74aceb21-8a9b-435f-be14-02669c8316d2
begin
	function multiply_connectivities(sbm::SBM, factor::Real)
		return SBM(sbm.sizes, factor .* sbm.connectivities)
	end
	
	function multiply_connectivities(csbm::CSBM, factor::Real)
		return CSBM(multiply_connectivities(csbm.sbm, factor), csbm.features)
	end
end

# ╔═╡ 4af3989f-78b9-4216-b85a-2af684af0a02
let
	σ_fast = 1
	σ_slow = 1

	csbm_fast = CSBM(
		SBM(100, 2, 1.0, 0.5),
		[
			MultivariateNormal(
				SVector(-100.0, -1.0),
				σ_fast^2 * SMatrix{2,2}(Diagonal(ones(2)))
			),
			MultivariateNormal(
				SVector(-100.0, +1.0),
				σ_fast^2 * SMatrix{2,2}(Diagonal(ones(2)))
			),
		]
	)

	csbm_slow = CSBM(
		SBM(100, 2, 0.05, 0.01),
		[
			MultivariateNormal(
				SVector(+100.0, -1.0),
				σ_slow^2 * SMatrix{2,2}(Diagonal(ones(2)))
			),
			MultivariateNormal(
				SVector(+100.0, +1.0),
				σ_slow^2 * SMatrix{2,2}(Diagonal(ones(2)))
			),
		]
	)

	csbm = reduce(vcat, [csbm_fast, csbm_slow])

	nb_layers = 10
 	nb_trajectories=10
	nb_graphs=10
	nb_samples=100

	errors_fast = random_walk_errors(rng, csbm_fast; nb_layers, nb_trajectories, nb_graphs, nb_samples)
	errors_slow = random_walk_errors(rng, csbm_slow; nb_layers, nb_trajectories, nb_graphs, nb_samples)
	errors = random_walk_errors(rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples)

	fig = Figure(size=(500, 500))
	ax_fast = Axis(fig[1, 1], title="fast")
	ax_slow = Axis(fig[1, 2], title="slow")
	ax_concat = Axis(fig[2, 1:2], title="concat")
	linkyaxes!(ax_slow, ax_fast)
	linkyaxes!(ax_concat, ax_fast)
	
	scatterlines!(ax_fast, 0:nb_layers, pmean.(errors_fast))
	scatterlines!(ax_slow, 0:nb_layers, pmean.(errors_slow))
	scatterlines!(ax_concat, 0:nb_layers, pmean.(errors))

	errorbars!(ax_fast, 0:nb_layers, pmean.(errors_fast), pstd.(errors_fast))
	errorbars!(ax_slow, 0:nb_layers, pmean.(errors_slow), pstd.(errors_slow))
	errorbars!(ax_concat, 0:nb_layers, pmean.(errors), pstd.(errors))
	fig
end

# ╔═╡ Cell order:
# ╟─9733cd58-fbee-4c0e-839d-15fc362c9abf
# ╠═10fa26e0-58a9-11ef-1536-e9fc7dc3721e
# ╠═652e47fb-56d1-4afe-afea-6f551ec39346
# ╠═7254e680-5fb8-424b-ac0c-db98e3724526
# ╟─38ddf907-2f83-4a43-89bb-f65356b6445b
# ╟─3fadb088-ef57-47c0-b564-8a2d268b514a
# ╠═a7e26ada-e581-4ef6-91ca-f0dba906ebb8
# ╟─67987f14-d637-4f33-b3e9-91597290cb74
# ╠═0735dfc4-85c0-491b-8bb6-58aa4272b772
# ╠═4d6d9739-baf0-4e65-8c86-4edaa50ceda0
# ╠═5839e3f7-baf4-4b75-b401-373d1cf83e73
# ╟─3c9cb7de-33eb-4422-ae50-95d5bf7484e0
# ╠═09ef5298-50fa-40f5-978d-24c8db4ff6e9
# ╠═efa56462-c94c-4b21-96e7-88ac5cfa9be1
# ╟─3a161f37-16ec-4604-b959-7104c4535b7e
# ╠═38202eb9-7103-412a-9ccd-3d8ca69bb71e
# ╟─7c6d090b-5d49-4a7e-adc3-be9028a74097
# ╠═220429bd-54d3-4c63-9540-cc648d8e9167
# ╟─ec86695e-aea0-4176-9de3-03b859deec39
# ╠═60bde353-26d3-408f-8e78-31e6ec26cd6a
# ╟─4c090dd7-f296-4836-bee4-0ed43a3ea7ec
# ╠═9ba113e6-b1bf-4f05-a185-2b6e77f453ab
# ╟─58887711-76ea-4c2e-b55d-6b00f8c1f93a
# ╠═74aceb21-8a9b-435f-be14-02669c8316d2
# ╠═4af3989f-78b9-4216-b85a-2af684af0a02
