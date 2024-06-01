degree_matrix(A) = Diagonal(Vector(map(sum, eachcol(A))))

function convolution!(X, X_scratch, A, D)
    mul!(X_scratch, A, X)
    X_scratch .+= X
    ldiv!(X, D, X_scratch)
    return nothing
end

function single_graph_embeddings(
    rng::AbstractRNG,
    graph::AbstractRandomGraph,
    features::Vector{<:MultivariateDistribution};
    nb_layers::Integer,
    resample_graph=false,
)
    X = stack(1:nb_vertices(graph); dims=1) do v
        c = community_of_vertex(graph, v)
        rand(rng, features[c])
    end
    A = rand(rng, graph)
    D = degree_matrix(A) + I
    H = copy(X)
    H_scratch = copy(H)
    H_history = Origin(0)([copy(H)])
    for l in 1:nb_layers
        if resample_graph
            A = rand(rng, graph)
            D = degree_matrix(A) + I
        end
        convolution!(H, H_scratch, A, D)
        push!(H_history, copy(H))
    end
    return H_history
end

function embeddings(
    rng::AbstractRNG,
    graph::AbstractRandomGraph,
    features::Vector{<:MultivariateDistribution};
    nb_layers::Integer,
    nb_graphs::Integer=1,
    resample_graph=false,
)
    H_histories = map(1:nb_graphs) do _
        single_graph_embeddings(rng, graph, features; nb_layers, resample_graph)
    end
    H_history = Origin(0)([
        permutedims(stack((H_history[l] for H_history in H_histories); dims=3), (3, 1, 2))  #
        for l in eachindex(first(H_histories))
    ])
    return H_history
end

function split_by_community(H::AbstractMatrix, graph::AbstractRandomGraph)
    H_split = map(1:nb_communities(graph)) do c
        H[community_range(graph, c), :]
    end
    return H_split
end

function split_by_community(H::AbstractArray{<:Any,3}, graph::AbstractRandomGraph)
    H_split = map(1:nb_communities(graph)) do c
        H[:, community_range(graph, c), :]
    end
    return H_split
end
