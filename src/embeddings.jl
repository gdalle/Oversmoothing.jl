degree_matrix(A) = Diagonal(Vector(map(sum, eachrow(A))))

function convolution!(X, X_scratch, A, D_plus_I)
    mul!(X_scratch, A, X)
    X_scratch .+= X
    ldiv!(D_plus_I, X_scratch)
    return copyto!(X, X_scratch)
end

function embeddings(
    rng::AbstractRNG,
    graph::AbstractRandomGraph,
    features::Vector{<:MultivariateDistribution};
    layers::Integer,
    resample_graph=false,
)
    X = stack(1:nb_vertices(graph); dims=1) do v
        c = community_of_vertex(graph, v)
        rand(rng, features[c])
    end
    A = rand(rng, graph)
    D_plus_I = degree_matrix(A) + I
    H = copy(X)
    H_scratch = copy(H)
    for l in 1:layers
        if resample_graph
            A = rand(rng, graph)
            D_plus_I = degree_matrix(A) + I
        end
        convolution!(H, H_scratch, A, D_plus_I)
    end
    return H
end

function split_by_community(H::AbstractMatrix, graph::AbstractRandomGraph)
    H_split = map(1:nb_communities(graph)) do c
        H[community_range(graph, c), :]
    end
    return H_split
end
