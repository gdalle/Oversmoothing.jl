function embeddings(
    rng::AbstractRNG,
    csbm::CSBM{C};
    nb_layers::Integer,
    nb_samples::Integer=1,
) where {C}
    (; sbm, features) = csbm
    S = nb_samples
    L = nb_layers
    N = nb_vertices(sbm)
    P = length(features[1])
    T = eltype(features[1])

    H_history = Array{T,4}(undef, S, L + 1, N, P)
    for s in 1:S
        X = stack(1:N; dims=1) do v
            c = community_of_vertex(sbm, v)
            rand(rng, features[c])
        end
        copyto!(view(H_history, s, 1, :, :), X)
        A = rand(rng, sbm)
        W = random_walk(A)
        H = copy(X)
        H_scratch = copy(H)
        for l in 1:L
            mul!(H_scratch, W, H)
            copyto!(H, H_scratch)
            copyto!(view(H_history, s, l + 1, :, :), H)
        end
    end

    histograms = map(0:L) do l
        [
            reduce(vcat, [H_history[s, l + 1, community_range(sbm, c), :] for s in 1:S]) for c in 1:C
        ]
    end
    return histograms
end
