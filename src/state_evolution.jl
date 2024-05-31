function state_evolution(
    er::ER,
    features::Vector{<:MultivariateDistribution};
    nb_layers::Integer,
    max_neighbors=nb_vertices(er),
)
    (; N, q) = er
    π₁⁰ = Mixture([only(features)], [1.0])
    π = [(π₁⁰,)]
    for l in 2:(nb_layers + 1)
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
    nb_layers::Integer,
    max_neighbors=nb_vertices(sbm),
)
    @assert length(features) == nb_communities(sbm)
    (; S, Q) = sbm
    N₁, N₂ = S
    n₁₁_max = n₂₁_max = min(N₁, max_neighbors)
    n₁₂_max = n₂₂_max = min(N₂, max_neighbors)
    π₁⁰ = Mixture([features[1]], [1.0])
    π₂⁰ = Mixture([features[2]], [1.0])
    π = Origin(0)([(π₁⁰, π₂⁰)])
    for l in 1:nb_layers
        μ₁ˡ⁻¹, μ₂ˡ⁻¹ = mean(π[l - 1][1]), mean(π[l - 1][2])
        Σ₁ˡ⁻¹, Σ₂ˡ⁻¹ = cov(π[l - 1][1]), cov(π[l - 1][2])
        μ₁ˡ_comp = Origin(0, 0)([  #
            ((n₁₁ + 1) * μ₁ˡ⁻¹ + n₁₂ * μ₂ˡ⁻¹) / ((n₁₁ + 1) + n₁₂)  #
            for n₁₁ in 0:n₁₁_max, n₁₂ in 0:n₁₂_max
        ])
        μ₂ˡ_comp = Origin(0, 0)([  #
            (n₂₁ * μ₁ˡ⁻¹ + (n₂₂ + 1) * μ₂ˡ⁻¹) / (n₂₁ + (n₂₂ + 1))  #
            for n₂₁ in 0:n₂₁_max, n₂₂ in 0:n₂₂_max
        ])
        Σ₁ˡ_comp = Origin(0, 0)([  #
            ((n₁₁ + 1)^2 * Σ₁ˡ⁻¹ + n₁₂^2 * Σ₂ˡ⁻¹) / ((n₁₁ + 1) + n₁₂)^2  #
            for n₁₁ in 0:n₁₁_max, n₁₂ in 0:n₁₂_max
        ])
        Σ₂ˡ_comp = Origin(0, 0)([  #
            (n₂₁^2 * Σ₁ˡ⁻¹ + (n₂₂ + 1)^2 * Σ₂ˡ⁻¹) / (n₂₁ + (n₂₂ + 1))^2  #
            for n₂₁ in 0:n₂₁_max, n₂₂ in 0:n₂₂_max
        ])
        π₁ˡ_comp = Origin(0, 0)([  #
            MvNormal(μ₁ˡ_comp[n₁₁, n₁₂], Σ₁ˡ_comp[n₁₁, n₁₂])  #
            for n₁₁ in 0:n₁₁_max, n₁₂ in 0:n₁₂_max
        ])
        π₂ˡ_comp = Origin(0, 0)([  #
            MvNormal(μ₂ˡ_comp[n₂₁, n₂₂], Σ₂ˡ_comp[n₂₁, n₂₂])  #
            for n₂₁ in 0:n₂₁_max, n₂₂ in 0:n₂₂_max
        ])
        π₁ˡ_weights = Origin(0, 0)([  #
            pdf(Binomial(N₁, Q[1, 1]), n₁₁) * pdf(Binomial(N₂, Q[1, 2]), n₁₂)  #
            for n₁₁ in 0:n₁₁_max, n₁₂ in 0:n₁₂_max
        ])
        π₂ˡ_weights = Origin(0, 0)([  #
            pdf(Binomial(N₂, Q[2, 1]), n₂₁) * pdf(Binomial(N₂, Q[2, 2]), n₂₂)  #
            for n₂₁ in 0:n₂₁_max, n₂₂ in 0:n₂₂_max
        ])
        π₁ˡ = Mixture(vec(π₁ˡ_comp), vec(π₁ˡ_weights))
        π₂ˡ = Mixture(vec(π₂ˡ_comp), vec(π₂ˡ_weights))
        push!(π, (π₁ˡ, π₂ˡ))
    end
    return Origin(0)(π)
end
