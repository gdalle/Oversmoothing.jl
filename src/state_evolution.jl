function state_evolution(
    er::ER,
    features::Vector{<:MultivariateDistribution};
    layers::Integer,
    max_neighbors=nb_vertices(er),
)
    (; N, q) = er
    π₁⁰ = Mixture([only(features)], [1.0])
    π = [(π₁⁰,)]
    for l in 2:(layers + 1)
        μ₁ˡ⁻¹ = mean(π[l - 1][1])
        Σ₁ˡ⁻¹ = cov(π[l - 1][1])
        μ₁ˡ_comp = [μ₁ˡ⁻¹ for n in 1:max_neighbors]
        Σ₁ˡ_comp = [((n + 1) / (n + 1)^2) * Σ₁ˡ⁻¹ for n in 1:max_neighbors]
        π₁ˡ_comp = [MvNormal(μ₁ˡ_comp[n], Σ₁ˡ_comp[n]) for n in 1:max_neighbors]
        π₁ˡ_weights = [pdf(Binomial(N, q), n) for n in 1:max_neighbors]
        π₁ˡ = Mixture(π₁ˡ_comp, π₁ˡ_weights)
        push!(π, (π₁ˡ,))
    end
    return π
end

function state_evolution(
    sbm::SBM{2},
    features::Vector{<:MultivariateDistribution};
    layers::Integer,
    max_neighbors=nb_vertices(sbm),
)
    @assert length(features) == nb_communities(sbm)
    (; S, Q) = sbm
    N₁, N₂ = S
    π₁⁰ = Mixture([features[1]], [1.0])
    π₂⁰ = Mixture([features[2]], [1.0])
    π = [(π₁⁰, π₂⁰)]
    for l in 2:(layers + 1)
        μ₁ˡ⁻¹, μ₂ˡ⁻¹ = mean(π[l - 1][1]), mean(π[l - 1][2])
        Σ₁ˡ⁻¹, Σ₂ˡ⁻¹ = cov(π[l - 1][1]), cov(π[l - 1][2])
        μ₁ˡ_comp = [
            ((n₁₁ + 1) * μ₁ˡ⁻¹ + n₁₂ * μ₂ˡ⁻¹) / ((n₁₁ + 1) + n₁₂)  #
            for n₁₁ in 1:max_neighbors, n₁₂ in 1:max_neighbors
        ]
        μ₂ˡ_comp = [
            (n₂₁ * μ₁ˡ⁻¹ + (n₂₂ + 1) * μ₂ˡ⁻¹) / (n₂₁ + (n₂₂ + 1))  #
            for n₂₁ in 1:max_neighbors, n₂₂ in 1:max_neighbors
        ]
        Σ₁ˡ_comp = [
            ((n₁₁ + 1) * Σ₁ˡ⁻¹ + n₁₂ * Σ₂ˡ⁻¹) / abs2((n₁₁ + 1) + n₁₂)  #
            for n₁₁ in 1:max_neighbors, n₁₂ in 1:max_neighbors
        ]
        Σ₂ˡ_comp = [
            (n₂₁ * Σ₁ˡ⁻¹ + (n₂₂ + 1) * Σ₂ˡ⁻¹) / abs2(n₂₁ + (n₂₂ + 1))  #
            for n₂₁ in 1:max_neighbors, n₂₂ in 1:max_neighbors
        ]
        π₁ˡ_comp = [  #
            MvNormal(μ₁ˡ_comp[n₁₁, n₁₂], Σ₁ˡ_comp[n₁₁, n₁₂])  #
            for n₁₁ in 1:max_neighbors, n₁₂ in 1:max_neighbors
        ]
        π₂ˡ_comp = [  #
            MvNormal(μ₂ˡ_comp[n₂₁, n₂₂], Σ₂ˡ_comp[n₂₁, n₂₂])  #
            for n₂₁ in 1:max_neighbors, n₂₂ in 1:max_neighbors
        ]
        π₁ˡ_weights = [
            pdf(Binomial(N₁, Q[1, 1]), n₁₁) * pdf(Binomial(N₂, Q[1, 2]), n₁₂)  #
            for n₁₁ in 1:max_neighbors, n₁₂ in 1:max_neighbors
        ]
        π₂ˡ_weights = [
            pdf(Binomial(N₂, Q[2, 1]), n₂₁) * pdf(Binomial(N₂, Q[2, 2]), n₂₂)  #
            for n₂₁ in 1:max_neighbors, n₂₂ in 1:max_neighbors
        ]
        π₁ˡ = Mixture(vec(π₁ˡ_comp), vec(π₁ˡ_weights))
        π₂ˡ = Mixture(vec(π₂ˡ_comp), vec(π₂ˡ_weights))
        push!(π, (π₁ˡ, π₂ˡ))
    end
    return π
end
