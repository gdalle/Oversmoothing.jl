degree_matrix(A::AbstractMatrix) = Diagonal(map(sum, eachrow(A)))

prepare(X::AbstractVector{<:Real}) = X
unprepare(X::AbstractVector{<:Real}) = X
prepare(X::AbstractVector{<:AbstractVector}) = transpose(stack(X))
unprepare(X::AbstractMatrix) = collect(eachrow(X))

function embedding_samples(
    rng::AbstractRNG,
    contextual::Contextual,
    convolution::AbstractConvolution,
    layers::Integer;
    samples=100,
)
    H_samples = tmap(1:samples) do s
        (; A, X) = rand(rng, contextual)
        D = degree_matrix(A)
        H = prepare(copy(X))
        scratch = copy(H)
        for l in 1:layers
            apply!(H, scratch, A, D, convolution)
        end
        unprepare(H)
    end
    return H_samples
end

function split_by_community(contextual::Contextual, H_samples::Vector{<:Array})
    N = length(contextual)
    C = nb_communities(contextual)
    T = eltype(first(H_samples))

    H_samples_by_community = [
        sizehint!(T[], length(H_samples) * community_size(contextual, c)) for c in 1:C
    ]
    for H in H_samples
        for v in 1:N
            c = get_community(contextual, v)
            push!(H_samples_by_community[c], H[v])
        end
    end
    return H_samples_by_community
end
