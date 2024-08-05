degree_matrix(A) = Diagonal(Vector(map(sum, eachcol(A))))
degree_matrix(A::SparseMatrixCSC) = Diagonal(length.(nzrange.(Ref(A), axes(A, 2))))

sumdropdims(x; dims) = dropdims(sum(x; dims); dims)

function community_walk_probabilities(rng::AbstractRNG, sbm::SBM; nb_layers=1)
    inds(c) = community_range(sbm, c)
    C = nb_communities(sbm)
    A = rand(rng, sbm)
    D = degree_matrix(A)
    W = inv(D + I) * (A + I)
    R = W^nb_layers
    R2 = abs2.(R)
    s = [sumdropdims(R[:, inds(c1)]; dims=2)[inds(c0)] for c0 in 1:C, c1 in 1:C]
    t = [sumdropdims(R2[:, inds(c1)]; dims=2)[inds(c0)] for c0 in 1:C, c1 in 1:C]
    return s, t
end
