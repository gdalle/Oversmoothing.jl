function embeddings(rng::AbstractRNG, csbm::CSBM; nb_layers::Integer, nb_graphs::Integer=1)
    (; sbm, features) = csbm
    G = nb_graphs
    L = nb_layers
    N = nb_vertices(sbm)
    C = nb_communities(sbm)
    P = length(features[1])
    T = eltype(features[1])

    H_history = Array{T,4}(undef, G, L + 1, N, P)
    for g in 1:G
        X = stack(1:N; dims=1) do v
            c = community_of_vertex(sbm, v)
            rand(rng, features[c])
        end
        copyto!(view(H_history, g, 1, :, :), X)
        A = rand(rng, sbm)
        W = random_walk(A)
        H = copy(X)
        H_scratch = copy(H)
        for l in 1:L
            mul!(H_scratch, W, H)
            copyto!(H, H_scratch)
            copyto!(view(H_history, g, l + 1, :, :), H)
        end
    end

    histograms = map(Iterators.product(0:L, 1:C)) do (l, c)
        reduce(vcat, [H_history[g, l + 1, community_range(sbm, c), :] for g in 1:G])
    end
    return histograms
end
