degree_matrix(A::SparseMatrixCSC) = Diagonal(length.(nzrange.(Ref(A), axes(A, 2))))

function random_walk(A)
    D = degree_matrix(A)
    return inv(D + I) * (A + I)
end

sumdropdims(f, x; dims) = dropdims(sum(f, x; dims); dims)
sumdropdims(x; dims) = dropdims(sum(x; dims); dims)

function community_walk_probabilities(rng::AbstractRNG, sbm::SBM; nb_layers=1, nb_graphs=1)
    (; sizes) = sbm
    inds(c) = community_range(sbm, c)
    C = nb_communities(sbm)
    G = nb_graphs
    L = nb_layers
    N = sum(sizes)

    s = [fill(NaN, G, community_size(sbm, c0)) for l in 0:L, c0 in 1:C, c1 in 1:C]
    t = [fill(NaN, G, community_size(sbm, c0)) for l in 0:L, c0 in 1:C, c1 in 1:C]

    for g in 1:G
        A = rand(rng, sbm)
        W = random_walk(A)
        R = I(N)
        for l in 0:L
            R_summed_by_blocks = [sumdropdims(R[:, inds(c1)]; dims=2) for c1 in 1:C]
            R2_summed_by_blocks = [sumdropdims(abs2, R[:, inds(c1)]; dims=2) for c1 in 1:C]
            for c0 in 1:C, c1 in 1:C
                s[l + 1, c0, c1][g, :] .= R_summed_by_blocks[c1][inds(c0)]
                t[l + 1, c0, c1][g, :] .= R2_summed_by_blocks[c1][inds(c0)]
            end
            R = R * W
            yield()
        end
    end
    return s, t
end

function random_walk_mixtures(rng::AbstractRNG, csbm::CSBM, ; nb_layers=1, nb_graphs=1)
    (; sbm, features) = csbm
    (; sizes) = sbm
    N = sizes
    G = nb_graphs
    L = nb_layers
    C = nb_communities(sbm)
    μ0, Σ0 = mean.(features), cov.(features)
    s, t = community_walk_probabilities(rng, sbm; nb_layers, nb_graphs)

    mixtures = Matrix{Mixture}(undef, L + 1, C)
    for c0 in 1:C
        mixtures[1, c0] = Mixture([features[c0]], [1.0])
        for l in 1:L
            μ = sum(vec(s[l + 1, c0, c1]) .* Ref(μ0[c1]) for c1 in 1:C)
            Σ = sum(vec(t[l + 1, c0, c1]) .* Ref(Σ0[c1]) for c1 in 1:C)
            mixtures[l + 1, c0] = Mixture(MultivariateNormal.(μ, Σ))
        end
    end
    return mixtures
end
