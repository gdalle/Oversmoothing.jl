degree_matrix(A::SparseMatrixCSC) = Diagonal(length.(nzrange.(Ref(A), axes(A, 2))))

function random_walk(A)
    D = degree_matrix(A)
    return inv(D + I) * (A + I)
end

sumdropdims(x; dims) = dropdims(sum(x; dims); dims)

function community_walk_probabilities(rng::AbstractRNG, sbm::SBM; nb_layers=1, nb_samples=1)
    inds(c) = community_range(sbm, c)
    C = nb_communities(sbm)
    st_by_sample = map(1:nb_samples) do _
        A = rand(rng, sbm)
        W = random_walk(A)
        R = W^nb_layers
        R2 = abs2.(R)
        R_summed_by_blocks = [sumdropdims(R[:, inds(c1)]; dims=2) for c1 in 1:C]
        R2_summed_by_blocks = [sumdropdims(R2[:, inds(c1)]; dims=2) for c1 in 1:C]
        s = [R_summed_by_blocks[c1][inds(c0)] for c0 in 1:C, c1 in 1:C]
        t = [R2_summed_by_blocks[c1][inds(c0)] for c0 in 1:C, c1 in 1:C]
        return s, t
    end
    s = reduce((a, b) -> vcat.(a, b), first.(st_by_sample))
    t = reduce((a, b) -> vcat.(a, b), last.(st_by_sample))
    return s, t
end

function empirical_mixtures(rng::AbstractRNG, csbm::CSBM; nb_layers=1, nb_samples=1)
    (; sbm, features) = csbm
    C = nb_communities(sbm)
    if nb_layers == 0
        mixtures = [Mixture([features[c]], [1.0]) for c in 1:C]
    else
        μ, Σ = mean.(features), cov.(features)
        s, t = community_walk_probabilities(rng, sbm; nb_layers, nb_samples)
        mixtures = map(1:C) do c0
            M = length(s[c0, 1])
            μ_c0 = [sum(s[c0, c1][m] * μ[c1] for c1 in 1:C) for m in 1:M]
            Σ_c0 = [sum(t[c0, c1][m] * Σ[c1] for c1 in 1:C) for m in 1:M]
            Mixture(MultivariateNormal.(μ_c0, Σ_c0), fill(1 / M, M))
        end
    end
    return mixtures
end
