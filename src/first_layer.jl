function first_layer_mixtures(csbm::CSBM{C}; max_neighbors=nb_vertices(csbm.sbm)) where {C}
    (; sbm, features) = csbm
    (; S, Q) = sbm
    N = min.(S, max_neighbors)

    μ0 = mean.(features)
    Σ0 = cov.(features)

    w = [Float64[] for c in 1:C]
    μ = [eltype(μ0)[] for c in 1:C]
    Σ = [eltype(Σ0)[] for c in 1:C]

    w_aux = OffsetArray{Float64,3}(undef, C, C, 0:maximum(N))
    for c in 1:C, c1 in 1:C, k1 in 0:maximum(N)
        w_aux[c, c1, k1] = binompdf(S[c1], Q[c, c1], k1)
    end

    lims = ntuple(c -> 0:N[c], Val(C))
    for c in 1:C, k in Iterators.product(lims...)
        w_ck = prod(w_aux[c, c1, k[c1]] for c1 in 1:C)
        w_ck > eps() || continue
        μ_ck = sum((k[c1] + (c == c1)) .* μ0[c1] for c1 in 1:C) / (sum(k) + 1)
        Σ_ck = sum((k[c1] + (c == c1)) .* Σ0[c1] for c1 in 1:C) / (sum(k) + 1)^2
        push!(w[c], w_ck)
        push!(μ[c], μ_ck)
        push!(Σ[c], Σ_ck)
    end

    mixtures = [Mixture(MultivariateNormal.(μ[c], Σ[c]), w[c]) for c in 1:C]
    return compress.(mixtures)
end
