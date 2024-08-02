function first_layer_mixtures(csbm::CSBM{C}; max_neighbors=nb_vertices(csbm.sbm)) where {C}
    (; sbm, features) = csbm
    (; S, Q) = sbm
    N = min.(S, max_neighbors)

    μ0 = mean.(features)
    Σ0 = cov.(features)

    lims = ntuple(c -> 0:N[c], Val(C))
    w = OffsetArray{Float64,1 + C}(undef, C, lims...)
    μ = OffsetArray{eltype(μ0),1 + C}(undef, C, lims...)
    Σ = OffsetArray{eltype(Σ0),1 + C}(undef, C, lims...)

    for c in 1:C
        ind = Tuple(c .== 1:C)
        for k in Iterators.product(lims...)
            w[c, k...] = prod(binompdf(S[c1], Q[c, c1], k[c1]) for c1 in 1:C)
            k_ = k .+ ind
            μ[c, k...] = sum(k_[c1] .* μ0[c1] for c1 in 1:C) / sum(k_)
            Σ[c, k...] = sum(k_[c1] .* Σ0[c1] for c1 in 1:C) / sum(k_)^2
        end
    end

    mixtures = [
        Mixture(
            Vector(vec(MultivariateNormal.(selectdim(μ, 1, c), selectdim(Σ, 1, c)))),
            Vector(vec(selectdim(w, 1, c))),
        ) for c in 1:C
    ]
    return mixtures
end
