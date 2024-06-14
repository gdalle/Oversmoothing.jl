function state_evolution(
    sbm::SBM{2}, features::NTuple{2}; nb_layers::Integer, max_neighbors=nb_vertices(sbm)
)
    @assert length(features) == nb_communities(sbm)
    (; S, Q) = sbm
    N1_max, N2_max = S
    N1 = min(N1_max, max_neighbors)
    N2 = min(N2_max, max_neighbors)

    w = OffsetArray{Float64,3}(undef, 2, 0:N1, 0:N2)
    ww = OffsetArray{Float64,6}(undef, 2, 2, 0:N1, 0:N2, 0:N1, 0:N2)
    for c1 in 1:2, k11 in 0:N1, k12 in 0:N2
        w[c1, k11, k12] = binompdf(N1, Q[c1, 1], k11) * binompdf(N2, Q[c1, 2], k12)
    end
    for c1 in 1:2, c2 in 1:2, k11 in 0:N1, k12 in 0:N2, k21 in 0:N1, k22 in 0:N2
        ww[c1, c2, k11, k12, k21, k22] =
            binompdf(N1, Q[c1, 1], k11) *
            binompdf(N2, Q[c1, 2], k12) *
            binompdf(N1, Q[c2, 1], k21) *
            binompdf(N2, Q[c2, 2], k22)
    end

    μ0 = mean.(features)
    Σ0 = cov.(features)

    μ = OffsetArray{typeof(μ0[1]),4}(undef, 0:nb_layers, 2, 0:N1, 0:N2)
    Σ = OffsetArray{typeof(Σ0[1]),4}(undef, 0:nb_layers, 2, 0:N1, 0:N2)
    Γ = OffsetArray{typeof(Σ0[1]),7}(undef, 0:nb_layers, 2, 2, 0:N1, 0:N2, 0:N1, 0:N2)

    μ_agg = OffsetArray{typeof(μ0[1]),2}(undef, 0:(nb_layers - 1), 2)
    Γ_agg = OffsetArray{typeof(Σ0[1]),3}(undef, 0:(nb_layers - 1), 2, 2)

    for c1 in 1:2
        for k11 in 0:N1, k12 in 0:N2
            μ[0, c1, k11, k12] = μ0[c1]
            Σ[0, c1, k11, k12] = Σ0[c1]
        end
    end
    for c1 in 1:2, c2 in 1:2
        for k11 in 0:N1, k12 in 0:N2, k21 in 0:N1, k22 in 0:N2
            Γ[0, c1, c2, k11, k12, k21, k22] = μ0[c1] * μ0[c2]' + (c1 == c2) * Σ0[c1]
        end
    end

    for l in 1:nb_layers
        # aggregates
        for c1 in 1:2
            μ_agg[l - 1, c1] = zero(μ0[c1])
            for k11 in 0:N1, k12 in 0:N2
                μ_agg[l - 1, c1] += w[c1, k11, k12] * μ[l - 1, c1, k11, k12]
            end
        end
        for c1 in 1:2, c2 in 1:2
            Γ_agg[l - 1, c1, c2] = zero(Σ0[c1])
            for k11 in 0:N1, k12 in 0:N2, k21 in 0:N1, k22 in 0:N2
                Γ_agg[l - 1, c1, c2] +=
                    ww[c1, c2, k11, k12, k21, k22] * Γ[l - 1, c1, c2, k11, k12, k21, k22]
            end
        end

        # components
        for c1 in 1:2, k11 in 0:N1, k12 in 0:N2
            μ[l, c1, k11, k12] =
                (k11 * μ_agg[l - 1, 1] + k12 * μ_agg[l - 1, 2]) / (1e-5 + k11 + k12)
        end
        for c1 in 1:2, c2 in 1:2, k11 in 0:N1, k12 in 0:N2, k21 in 0:N1, k22 in 0:N2
            Γ[l, c1, c2, k11, k12, k21, k22] =
                (
                    k11 * k21 * Γ_agg[l - 1, 1, 1] +
                    k11 * k22 * Γ_agg[l - 1, 1, 2] +
                    k12 * k21 * Γ_agg[l - 1, 2, 1] +
                    k12 * k22 * Γ_agg[l - 1, 2, 2]
                ) / (1e-5 + k11 * k21 + k11 * k22 + k12 * k21 + k12 * k22)
        end
        for c1 in 1:2, k11 in 0:N1, k12 in 0:N2
            Σ[l, c1, k11, k12] = (
                Γ[l, c1, c1, k11, k12, k11, k12] -  #
                μ[l, c1, k11, k12] * μ[l, c1, k11, k12]'
            )
        end
    end

    p = Origin(0, 1)([
        Mixture( #
            vec(MultivariateNormal.(μ[l, c1, :, :], Σ[l, c1, :, :])),
            vec(w[c1, :, :]),
        ) for l in 0:nb_layers, c1 in 1:2
    ])
    return p
end
