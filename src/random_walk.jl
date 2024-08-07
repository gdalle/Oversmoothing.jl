degree_matrix(A::SparseMatrixCSC) = Diagonal(length.(nzrange.(Ref(A), axes(A, 2))))

function random_walk(A)
    D = degree_matrix(A)
    return inv(D + I) * (A + I)
end

sumdropdims(x; dims) = dropdims(sum(x; dims); dims)

function community_walk_probabilities(rng::AbstractRNG, sbm::SBM; nb_layers=1)
    inds(c) = community_range(sbm, c)
    C = nb_communities(sbm)
    A = rand(rng, sbm)
    W = random_walk(A)
    R = W^nb_layers
    R2 = abs2.(R)
    R_by_blocks = [R[inds(c0), inds(c1)] for c0 in 1:C, c1 in 1:C]
    R2_by_blocks = [R2[inds(c0), inds(c1)] for c0 in 1:C, c1 in 1:C]
    s = [sumdropdims(R_by_blocks[c0, c1]; dims=2) for c0 in 1:C, c1 in 1:C]
    t = [sumdropdims(R2_by_blocks[c0, c1]; dims=2) for c0 in 1:C, c1 in 1:C]
    return s, t
end

function empirical_mixtures(rng::AbstractRNG, csbm::CSBM{C}; nb_layers=1) where {C}
    (; sbm, features) = csbm

    μ = mean.(features)
    Σ = cov.(features)

    s, t = community_walk_probabilities(rng, sbm)

    mixtures = map(1:C) do c0
        M = length(s[c0, 1])
        μ_c0 = [sum(s[c0, c1][m] * μ[c1] for c1 in 1:C) for m in 1:M]
        Σ_c0 = [sum(t[c0, c1][m] * Σ[c1] for c1 in 1:C) for m in 1:M]
        Mixture(MultivariateNormal.(μ_c0, Σ_c0), fill(1 / M, M))
    end

    return mixtures
end
