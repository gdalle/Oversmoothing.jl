function state_evolution(
    sbm::SBM{2},
    features::NTuple{2,<:MultivariateDistribution};
    nb_layers::Integer,
    max_neighbors=nb_vertices(sbm),
)
    @assert length(features) == nb_communities(sbm)
    (; S, Q) = sbm
    N1, N2 = S
    N1_max = min(N1, max_neighbors)
    N2_max = min(N2, max_neighbors)
    p01 = MixtureModel([features[1]], Categorical([1.0]))
    p02 = MixtureModel([features[2]], Categorical([1.0]))
    p = Matrix{typeof(p01)}(undef, nb_layers + 1, 2)
    p[1, 1] = p01
    p[1, 2] = p02
    for l in 1:nb_layers
        μˡ⁻¹ = (mean(p[l, 1]), mean(p[l, 2]))
        Σˡ⁻¹ = (cov(p[l, 1]), cov(p[l, 2]))
        dist_type = typeof(MvNormal(sum(μˡ⁻¹), sum(Σˡ⁻¹)))
        for c in 1:2
            means = OffsetMatrix{eltype(μˡ⁻¹)}(undef, 0:N1_max, 0:N2_max)
            covs = OffsetMatrix{eltype(Σˡ⁻¹)}(undef, 0:N1_max, 0:N2_max)
            comps = OffsetMatrix{dist_type}(undef, 0:N1_max, 0:N2_max)
            weights = OffsetMatrix{eltype(Q)}(undef, 0:N1_max, 0:N2_max)
            n1_dist = Binomial(N1, Q[c, 1])
            n2_dist = Binomial(N2, Q[c, 2])
            for n1 in 0:N1_max, n2 in 0:N2_max
                means[n1, n2] =
                    ((n1 + (c == 1)) * μˡ⁻¹[1] + (n2 + (c == 2)) * μˡ⁻¹[2]) / (n1 + n2 + 1)
                covs[n1, n2] =
                    ((n1 + (c == 1)) * Σˡ⁻¹[1] + (n2 + (c == 2)) * Σˡ⁻¹[2]) /
                    (n1 + n2 + 1)^2
                comps[n1, n2] = MvNormal(means[n1, n2], covs[n1, n2])
                weights[n1, n2] = pdf(n1_dist, n1) * pdf(n2_dist, n2)
            end
            p[l + 1, c] = MixtureModel(vec(comps), Categorical(vec(weights)))
        end
    end
    return p
end
