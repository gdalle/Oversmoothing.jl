### A Pluto.jl notebook ###
# v0.19.46

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

# ╔═╡ a2f4d87b-1a9a-4e68-9af2-6492154dbcf3
md"""
## Theory
"""

# ╔═╡ 1d1a8351-5183-4c1e-84af-c9f14b6e7505
md"""
Lu and Sen (2023) consider a symmetric binary CSBM with $n$ nodes and $p$-dimensional features.
The edge probabilities are $a/n$ (intra-community) and $b/n$ (inter-community) respectively.
The Gaussian community features have variance $1$ and mean $\pm u \sqrt{\mu / n}$, where $u \sim \mathcal{N}(0, I_p)$.
The phase transition for detectability happens at
```math
\lambda^2 + \frac{\mu^2}{\gamma} > 1
```
where the parameters are $\gamma = \lim n/p$ and $\lambda$ defined by
```math
\begin{cases}
d = (a + b) / 2 \\
a = d + \lambda \sqrt{d} \\
b = d - \lambda \sqrt{d}
\end{cases}
\implies \lambda = \frac{a - b}{2 \sqrt{d}} = \frac{a-b}{\sqrt{2} \sqrt{a+b}}
```
"""

# ╔═╡ 3c9cb7de-33eb-4422-ae50-95d5bf7484e0
md"""
## Toy CSBMs
"""

# ╔═╡ c5a94b08-78dc-4dd6-96c2-7ebe488205d7
let
    C = 4
    csbm = LinearCSBM1d(; N=100, C=C, p_in=0.03, p_out=0.01, Δμ=10)

    fig = Figure()
    ax = Axis(fig[1, 1])
    for c in 1:C
        x = [rand(rng, csbm.features[c]) for _ in 1:100]
        hist!(ax, first.(x))
    end
    fig
end

# ╔═╡ c1a99529-cf54-4ff7-ae44-4bdfce781a07
let
    C = 8
    csbm = CircularCSBM2d(; N=200, C=C, p_in=0.03, p_out=0.01, Δμ=10)

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
        UnivariateNormal(-1.0, 0.02),
        UnivariateNormal(-0.0, 0.01),
        UnivariateNormal(+1.0, 0.005),
    ]
    csbm = CSBM(sbm, features)

    nb_layers = 2
    nb_graphs = 100
    embeddings = empirical_embeddings(rng, csbm; nb_layers, nb_graphs)
    densities = random_walk_densities(rng, csbm; nb_layers, nb_graphs)

    plot_1d(csbm, embeddings, densities; theme=MYTHEME)
end

# ╔═╡ 67987f14-d637-4f33-b3e9-91597290cb74
md"""
## 2D
"""

# ╔═╡ 0735dfc4-85c0-491b-8bb6-58aa4272b772
let
    sbm = SBM(300, 3, 0.03, 0.01)
    features = [
        BivariateNormal([-2.0, 0.0], [1.0 0.; 0. 2.0]),
        BivariateNormal([0.0, 2.0], [2.0 -0.4; -0.4 1.0]),
        BivariateNormal([+3.0, -1.0], [1.0 0.3; 0.3 1.0]),
    ]
    csbm = CSBM(sbm, features)

    nb_layers = 2
    nb_graphs = 100
    embeddings = empirical_embeddings(rng, csbm; nb_layers, nb_graphs)
    densities = random_walk_densities(rng, csbm; nb_layers, nb_graphs)

    plot_2d(csbm, embeddings, densities; theme=MYTHEME)
end

# ╔═╡ f53e238d-6f08-4da0-a5af-7278a7c64e5c
md"""
# First layer
"""

# ╔═╡ aed0b4be-36c4-412d-a9a0-d0553e67cc61
md"""
## Phase transition
"""

# ╔═╡ f57cf269-d9a4-4393-8a21-ff898881c1ee
@kwdef struct PhaseTransitionExperiment
	N
	C
	d
	λ2_vals
	Δμ2_vals
	accuracy0_vals
	accuracy1_vals
end

# ╔═╡ 8aafdda1-9730-4954-902c-c22669503f91
phase_transition_experiment = let
	N = 100
    C = 2
	d = 5
	
	λ2_vals = 0.05:0.02:1.0
	Δμ2_vals = 0.05:0.02:2.0
    accuracy0_vals = fill(NaN, length(λ2_vals), length(Δμ2_vals))
    accuracy1_vals = fill(NaN, length(λ2_vals), length(Δμ2_vals))

    @progress for i in eachindex(λ2_vals), j in eachindex(Δμ2_vals)
		λ, Δμ = sqrt(λ2_vals[i]), sqrt(Δμ2_vals[j])
		a = d + λ * sqrt(d)
		b = d - λ * sqrt(d)
        csbm = LinearCSBM1d(; N, C, p_in=a/N, p_out=b/N, Δμ)

		accuracy0_vals[i, j] = accuracy_zeroth_layer(csbm; rtol=1e-5) |> value
		accuracy1_vals[i, j] = accuracy_first_layer(csbm; rtol=1e-5) |> value
    end

    PhaseTransitionExperiment(;
		N, C, d, λ2_vals, Δμ2_vals, accuracy0_vals, accuracy1_vals
	)
end

# ╔═╡ 5c6f6a76-378e-4687-82ab-40352f935a58
let
	(; λ2_vals, Δμ2_vals, accuracy0_vals, accuracy1_vals) = phase_transition_experiment
    acc_diff_vals = accuracy1_vals .- accuracy0_vals

    fig = Figure(size=(500, 500))
    ax = Axis(fig[1, 1]; aspect=1, xlabel=L"\lambda^2", ylabel=L"(\Delta \mu)^2")
    hm = heatmap!(
        ax,
        λ2_vals,
        Δμ2_vals,
        acc_diff_vals;
        colormap=Reverse(:curl),
        colorrange=(-maximum(abs, acc_diff_vals), maximum(abs, acc_diff_vals)),
    )
    Colorbar(fig[1, 2], hm; label=L"accuracy improvement after $1$ layer")
	rowsize!(fig.layout, 1, Aspect(1, 1))
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
	Δμ
	accuracy0_vals
	accuracy1_vals
end

# ╔═╡ c5ddce4c-21ae-4eb9-a411-a576ac8f766d
connectivity_experiment = let
	N = 100
    C = 2
    Δμ = 1.0

    p_in_vals = (0:0.2:10) ./ N
    p_out_vals = (0:0.2:10) ./ N

	accuracy0_vals = fill(NaN, length(p_in_vals), length(p_out_vals))
    accuracy1_vals = fill(NaN, length(p_in_vals), length(p_out_vals))

    @progress for i in eachindex(p_in_vals), j in eachindex(p_out_vals)
        p_in, p_out = p_in_vals[i], p_out_vals[j]
        csbm = LinearCSBM1d(; N, C, p_in, p_out, Δμ)
        accuracy0_vals[i, j] = accuracy_zeroth_layer(csbm; rtol=1e-4) |> value
        accuracy1_vals[i, j] = accuracy_first_layer(csbm; rtol=1e-4) |> value
    end
    
	ConnectivityExperiment(;
		N, C, p_in_vals, p_out_vals, Δμ, accuracy0_vals, accuracy1_vals
	)
end

# ╔═╡ 2b272855-8b04-47d0-b2c5-c672ab633f79
let
	(; p_in_vals, p_out_vals, accuracy0_vals, accuracy1_vals) = connectivity_experiment

	acc_diff_vals = accuracy1_vals .- accuracy0_vals

    fig = Figure(size=(500, 500))
    ax = Axis(fig[1, 1]; aspect=1, xlabel=L"p_{in}", ylabel=L"p_{out}")
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
	Δμ
	accuracies_rw
	accuracies_lr
end

# ╔═╡ 9ba113e6-b1bf-4f05-a185-2b6e77f453ab
oscillations_experiments = map(2:5) do C
	@info "C = $C"
	N = 120
	p_in = 0.05
	p_out = 0.01
	Δμ = 2
	
	nb_layers, nb_trajectories, nb_graphs = 6, 20, 10
    
	csbm = CircularCSBM2d(; N, C, p_in, p_out, Δμ)

    accuracies_rw = accuracy_by_depth(
		rng, csbm, Val(:randomwalk);
		nb_layers, nb_trajectories, nb_graphs
	)
	accuracies_lr = accuracy_by_depth(
		rng, csbm, Val(:logisticregression);
		nb_layers, nb_trajectories, nb_graphs
	)

    OscillationsExperiment(; N, C, p_in, p_out, Δμ, accuracies_rw, accuracies_lr)
end

# ╔═╡ a52d2a3a-4e6a-40a9-90ab-543d3e021f60
let
	fig = Figure(size=(800, 300))

	axref = nothing

	for (j, oscillations_experiment) in enumerate(oscillations_experiments)
		(; C, accuracies_rw, accuracies_lr) = oscillations_experiment
		nb_layers = length(accuracies_rw) - 1
	
		val_rw, unc_rw = value.(accuracies_rw), uncertainty.(accuracies_rw)
		val_lr, unc_lr = value.(accuracies_lr), uncertainty.(accuracies_lr)

		if j == 1
	    	ax = Axis(fig[1, j]; xlabel="depth", ylabel="accuracy", title=L"C = %$C")
			axref = ax
		else
	    	ax = Axis(fig[1, j]; xlabel="depth", title=L"C = %$C")
		end
			
		band!(ax, 0:nb_layers, val_lr - unc_lr, val_lr + unc_lr, label="logistic regression", alpha=0.3)
	    scatterlines!(ax, 0:nb_layers, val_rw, label="theory")
	    errorbars!(ax, 0:nb_layers, val_rw, unc_rw)
	end

	Legend(fig[2, 1:end], axref, orientation=:horizontal)
    fig
end

# ╔═╡ eb3eb315-4c78-46d8-9256-dac0c1e4ebad
md"""
## Optimal depth
"""

# ╔═╡ d2ce838b-776f-4221-be50-90a5e6b7c34b
@kwdef struct OptimalDepthExperiment
	N
	C
	p_in_vals
	p_out
	Δμ
	accuracies
end

# ╔═╡ a0b6dc2d-3df5-43f4-a5eb-568a796707bb
optimal_depth_experiments = map(2:5) do C
	N = 120
	p_in_vals = reverse(0.01:0.01:0.05)
	p_out = 0.01
	Δμ = 1
	
	nb_layers, nb_trajectories, nb_graphs = 6, 10, 10
	
	accuracies = Vector{Any}(undef, length(p_in_vals))
	@progress "C = $C" for i in eachindex(p_in_vals)
		p_in = p_in_vals[i]
		csbm = CircularCSBM2d(; N, C, p_in, p_out, Δμ)
		accuracies[i] = accuracy_by_depth(
			rng, csbm, Val(:randomwalk);
			nb_layers, nb_trajectories, nb_graphs
		)
	end

	OptimalDepthExperiment(; N, C, p_in_vals, p_out, Δμ, accuracies)
end

# ╔═╡ c1fcb1d2-f4b6-445a-962d-5ce9bd8271d1
let
	fig = Figure(size=(1000, 400))

	axref = nothing
	
	for (j, optimal_depth_experiment) in enumerate(optimal_depth_experiments)
		(; C, p_in_vals, accuracies) = optimal_depth_experiment
		nb_layers = length(accuracies[1]) - 1
		
		if j == 1
			ax = Axis(fig[1, j]; xlabel="depth", ylabel="accuracy", title=L"C = %$C")
			axref = ax
		else
			ax = Axis(fig[1, j]; xlabel="depth", title=L"C = %$C")
		end

		for (i, p_in) in enumerate(p_in_vals)
			scatterlines!(ax, 0:nb_layers, value.(accuracies[i]), label=L"p_{in} = %$p_in")
	    	errorbars!(ax, 0:nb_layers, value.(accuracies[i]), uncertainty.(accuracies[i]))
		end
	end

	Legend(fig[2, 1:end], axref, orientation=:horizontal)
	fig
end

# ╔═╡ Cell order:
# ╟─d46ca486-6b19-4b00-b996-2762d683eb1e
# ╟─9733cd58-fbee-4c0e-839d-15fc362c9abf
# ╠═10fa26e0-58a9-11ef-1536-e9fc7dc3721e
# ╠═652e47fb-56d1-4afe-afea-6f551ec39346
# ╠═8b80f3b5-4fdd-48c8-9f0d-52e35535a653
# ╟─a2f4d87b-1a9a-4e68-9af2-6492154dbcf3
# ╟─1d1a8351-5183-4c1e-84af-c9f14b6e7505
# ╟─3c9cb7de-33eb-4422-ae50-95d5bf7484e0
# ╠═c5a94b08-78dc-4dd6-96c2-7ebe488205d7
# ╠═c1a99529-cf54-4ff7-ae44-4bdfce781a07
# ╟─38ddf907-2f83-4a43-89bb-f65356b6445b
# ╟─3fadb088-ef57-47c0-b564-8a2d268b514a
# ╠═a7e26ada-e581-4ef6-91ca-f0dba906ebb8
# ╟─67987f14-d637-4f33-b3e9-91597290cb74
# ╠═0735dfc4-85c0-491b-8bb6-58aa4272b772
# ╟─f53e238d-6f08-4da0-a5af-7278a7c64e5c
# ╟─aed0b4be-36c4-412d-a9a0-d0553e67cc61
# ╠═f57cf269-d9a4-4393-8a21-ff898881c1ee
# ╠═8aafdda1-9730-4954-902c-c22669503f91
# ╠═5c6f6a76-378e-4687-82ab-40352f935a58
# ╟─4b8d0758-7c36-42d4-b2c7-df5a4b033d38
# ╠═4b7d7234-ece3-4875-a232-562ebe418055
# ╠═c5ddce4c-21ae-4eb9-a411-a576ac8f766d
# ╠═2b272855-8b04-47d0-b2c5-c672ab633f79
# ╟─9a881c91-fffb-416a-beb9-ad5985063273
# ╟─4c090dd7-f296-4836-bee4-0ed43a3ea7ec
# ╠═539f2393-3ba8-4287-8e82-aa602e76cf01
# ╠═9ba113e6-b1bf-4f05-a185-2b6e77f453ab
# ╠═a52d2a3a-4e6a-40a9-90ab-543d3e021f60
# ╟─eb3eb315-4c78-46d8-9256-dac0c1e4ebad
# ╠═d2ce838b-776f-4221-be50-90a5e6b7c34b
# ╠═a0b6dc2d-3df5-43f4-a5eb-568a796707bb
# ╠═c1fcb1d2-f4b6-445a-962d-5ce9bd8271d1
