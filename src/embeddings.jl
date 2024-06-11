degree_matrix(A) = Diagonal(Vector(map(sum, eachcol(A))))

function convolution!(X, X_scratch, A, D)
    mul!(X_scratch, A, X)
    X_scratch .+= X
    ldiv!(X, D, X_scratch)
    return nothing
end

function embeddings(
    rng::AbstractRNG,
    sbm::StochasticBlockModel{C},
    features::NTuple{C};
    nb_layers::Integer,
    nb_graphs::Integer=1,
    resample_graph=false,
) where {C}
    T = eltype(first(features))
    H_history = Array{T,4}(
        undef, nb_graphs, nb_layers + 1, nb_vertices(sbm), length(first(features))
    )
    for g in 1:nb_graphs
        X = stack(1:nb_vertices(sbm); dims=1) do v
            c = community_of_vertex(sbm, v)
            rand(rng, features[c])
        end
        copyto!(view(H_history, g, 1, :, :), X)
        A = rand(rng, sbm)
        D = degree_matrix(A) + I
        H = copy(X)
        H_scratch = copy(H)
        for l in 1:nb_layers
            if resample_graph
                A = rand(rng, sbm)
                D = degree_matrix(A) + I
            end
            convolution!(H, H_scratch, A, D)
            copyto!(view(H_history, g, l + 1, :, :), H)
        end
    end
    return H_history
end

function split_by_community(H_history::AbstractArray{<:Real,3}, sbm::SBM)
    H_history_split = map(1:nb_communities(sbm)) do c
        H_history[:, community_range(sbm, c), :]
    end
    return H_history_split
end

function split_by_community(H_history::AbstractArray{<:Real,4}, sbm::SBM)
    H_history_split = map(1:nb_communities(sbm)) do c
        H_history[:, :, community_range(sbm, c), :]
    end
    return H_history_split
end
