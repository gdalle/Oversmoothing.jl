
prepare(X::AbstractVector{<:Real}) = X
unprepare(X::AbstractVector{<:Real}) = X
prepare(X::AbstractVector{<:AbstractVector}) = transpose(stack(X))
unprepare(X::AbstractMatrix) = collect(eachrow(X))

function embedding_samples(
    rng::AbstractRNG,
    contextual::Contextual,
    ::Type{Convolution},
    layers::Integer;
    samples=100,
) where {Convolution<:AbstractConvolution}
    H_samples = tmap(1:samples) do s
        (; A, X) = rand(rng, contextual)
        conv = Convolution(A)
        H = prepare(copy(X))
        scratch = copy(H)
        for l in 1:layers
            apply!(H, scratch, conv)
            H .= scratch
        end
        return unprepare(H)
    end
    return H_samples
end

function embedding_samples_indep(
    rng::AbstractRNG,
    contextual::Contextual,
    ::Type{Convolution},
    layers::Integer;
    samples=100,
) where {Convolution<:AbstractConvolution}
    graph = contextual.graph
    H_samples = tmap(1:samples) do s
        (; X) = rand(rng, contextual)
        iterated_conv_indep = [
            single_coeff(Convolution(rand(rng, graph)), layers, u, v) for
            u in 1:length(graph), v in 1:length(graph)
        ]
        H = prepare(copy(X))
        H = iterated_conv_indep * H
        return unprepare(H)
    end
    return H_samples
end

function split_by_community(graph::AbstractRandomGraph, H_samples::Vector{<:Array})
    N = length(graph)
    C = nb_communities(graph)
    T = eltype(first(H_samples))

    H_samples_by_community = [
        sizehint!(T[], length(H_samples) * community_size(graph, c)) for c in 1:C
    ]
    for H in H_samples
        for v in 1:N
            c = get_community(graph, v)
            push!(H_samples_by_community[c], H[v])
        end
    end
    return H_samples_by_community
end
