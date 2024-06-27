function first_layer_mixtures(
    sbm::SBM{2}, features::NTuple{2}; max_neighbors=nb_vertices(sbm)
)
    (; S, Q) = sbm
    N1_max, N2_max = S
    N1 = min(N1_max, max_neighbors)
    N2 = min(N2_max, max_neighbors)

    μ0 = mean.(features)
    Σ0 = cov.(features)
    Γ0 = [Σ0[c] + μ0[c] * μ0[c]' for c in 1:2]

    w = OffsetArray{Float64,3}(undef, 2, 0:N1, 0:N2)
    μ = OffsetArray{eltype(μ0),3}(undef, 2, 0:N1, 0:N2)
    Γ = OffsetArray{eltype(Σ0),3}(undef, 2, 0:N1, 0:N2)
    Σ = OffsetArray{eltype(Σ0),3}(undef, 2, 0:N1, 0:N2)

    for c in 1:2, k1 in 0:N1, k2 in 0:N2
        w[c, k1, k2] = binompdf(N1_max, Q[c, 1], k1) * binompdf(N2_max, Q[c, 2], k2)
        μ[c, k1, k2] = ((k1 + (c == 1)) * μ0[1] + (k2 + (c == 2)) * μ0[2]) / (k1 + k2 + 1)
        Γ[c, k1, k2] = ((k1 + (c == 1)) * Γ0[1] + (k2 + (c == 2)) * Γ0[2]) / (k1 + k2 + 1)^2
        Σ[c, k1, k2] = Γ[c, k1, k2] - μ[c, k1, k2] * μ[c, k1, k2]'
        @show Γ[c, k1, k2], μ[c, k1, k2], Σ[c, k1, k2]
    end

    p = [
        Mixture( #
            vec(MultivariateNormal.(μ[c, :, :], Σ[c, :, :])),
            vec(w[c, :, :]),
        ) for c in 1:2
    ]
    return p
end
