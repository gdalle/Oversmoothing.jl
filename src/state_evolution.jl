function state_evolution(
    er::ER,
    features::Vector{<:MultivariateDistribution};
    layers::Integer,
    max_neighbors=nb_vertices(er),
)
    (; N, q) = er
    π1 = [Mixture([only(features)], [1.0])]
    for l in 2:(layers + 1)
        μ_prev = mean(π1[l - 1])
        Σ_prev = cov(π1[l - 1])
        μs = [μ_prev for n in 1:max_neighbors]
        Σs = [((n + 1) / (n + 1)^2) * Σ_prev for n in 1:max_neighbors]
        components = [MvNormal(μs[n], Σs[n]) for n in 1:max_neighbors]
        weights = [pdf(Binomial(N, q), n) for n in 1:max_neighbors]
        push!(π1, Mixture(components, weights))
    end
    return π1
end
