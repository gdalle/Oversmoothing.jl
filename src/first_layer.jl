function first_layer_densities(csbm::CSBM{C}; max_neighbors=nb_vertices(csbm.sbm)) where {C}
    (; sbm, features) = csbm
    (; sizes, connectivities) = sbm
    N = min.(sizes, max_neighbors)

    μ0 = mean.(features)
    Σ0 = cov.(features)

    w = [Float64[] for c0 in 1:C]
    μ = [eltype(μ0)[] for c0 in 1:C]
    Σ = [eltype(Σ0)[] for c0 in 1:C]

    w_aux = Array{Float64,3}(undef, C, C, maximum(N) + 1)
    for c0 in 1:C, c1 in 1:C, k1 in 0:maximum(N)
        w_aux[c0, c1, k1 + 1] = binompdf(sizes[c1], connectivities[c0, c1], k1)
    end

    lims = ntuple(c -> 0:N[c], Val(C))
    for c0 in 1:C, k in Iterators.product(lims...)
        w_ck = prod(w_aux[c0, c1, k[c1] + 1] for c1 in 1:C)
        w_ck > eps() || continue
        μ_ck = sum((k[c1] + (c0 == c1)) .* μ0[c1] for c1 in 1:C) / (sum(k) + 1)
        Σ_ck = sum((k[c1] + (c0 == c1)) .* Σ0[c1] for c1 in 1:C) / (sum(k) + 1)^2
        push!(w[c0], w_ck)
        push!(μ[c0], μ_ck)
        push!(Σ[c0], Σ_ck)
    end

    densities = [Mixture(MultivariateNormal.(μ[c0], Σ[c0]), w[c0]) for c0 in 1:C]
    return densities
end
