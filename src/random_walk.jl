degree_matrix(A) = Diagonal(Vector(map(sum, eachcol(A))))
degree_matrix(A::SparseMatrixCSC) = Diagonal(length.(nzrange.(Ref(A), axes(A, 2))))

sumdropdims(x; dims) = dropdims(sum(x; dims); dims)

function community_walk_probabilities(rng::AbstractRNG, sbm::SBM; nb_layers=1)
    C = nb_communities(sbm)
    A = rand(rng, sbm)
    D = degree_matrix(A)
    W = (D + I) \ (A + I)
    R = W^nb_layers
    R2 = R .^ 2
    r = [sumdropdims(R[:, community_range(sbm, c1)]; dims=2) for c1 in 1:C]
    s = [sumdropdims(R2[:, community_range(sbm, c1)]; dims=2) for c1 in 1:C]
    return (
        [r[c1][community_range(sbm, c0)] for c0 in 1:C, c1 in 1:C],
        [s[c1][community_range(sbm, c0)] for c0 in 1:C, c1 in 1:C],
    )
end
