function first_layer_mixtures(
    sbm::SBM{2}, features::NTuple{2}; max_neighbors=nb_vertices(sbm)
)
    (; S, Q) = sbm
    N1_max, N2_max = S
    N1 = min(N1_max, max_neighbors)
    N2 = min(N2_max, max_neighbors)

    μ0 = mean.(features)
    Σ0 = cov.(features)
    Γ0_same = [Σ0[c] + μ0[c] * μ0[c]' for c in 1:2]  # same vertex
    Γ0_diff = [μ0[c1] * μ0[c2]' for c1 in 1:2, c2 in 1:2]  # different vertex

    w = OffsetArray{Float64,3}(undef, 2, 0:N1, 0:N2)
    μ = OffsetArray{eltype(μ0),3}(undef, 2, 0:N1, 0:N2)
    Γ = OffsetArray{eltype(Σ0),3}(undef, 2, 0:N1, 0:N2)
    Σ = OffsetArray{eltype(Σ0),3}(undef, 2, 0:N1, 0:N2)

    for c in 1:2, k1 in 0:N1, k2 in 0:N2
        w[c, k1, k2] = binompdf(N1_max, Q[c, 1], k1) * binompdf(N2_max, Q[c, 2], k2)
        k1_ = k1 + (c == 1)
        k2_ = k2 + (c == 2)
        μ[c, k1, k2] = (k1_ * μ0[1] + k2_ * μ0[2]) / (k1_ + k2_)
        Γ[c, k1, k2] =
            (
                (k1_ * k2_ * Γ0_diff[1, 2] + k2_ * k1_ * Γ0_diff[2, 1]) +
                (k1_ * (k1_ - 1) * Γ0_diff[1, 1] + k2_ * (k2_ - 1) * Γ0_diff[2, 2]) +
                (k1_ * Γ0_same[1] + k2_ * Γ0_same[2])
            ) / (k1_ + k2_)^2
        Σ[c, k1, k2] = Γ[c, k1, k2] - μ[c, k1, k2] * μ[c, k1, k2]'
    end

    p = [
        Mixture( #
            vec(MultivariateNormal.(μ[c, :, :], Σ[c, :, :])),
            vec(w[c, :, :]),
        ) for c in 1:2
    ]
    return p
end
