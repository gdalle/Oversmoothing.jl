function unnest_matrix_of_blocks(matrix_of_blocks::AbstractMatrix{<:AbstractMatrix})
    vector_of_blocks = map(Base.splat(hcat), eachrow(matrix_of_blocks))
    return vcat(vector_of_blocks...)
end

struct StochasticBlockModel{C,T<:Real}
    sizes::Vector{Int}
    connectivities::Matrix{T}

    function StochasticBlockModel(
        sizes::AbstractVector{Int}, connectivities::AbstractMatrix{T}
    ) where {T}
        C = length(sizes)
        @assert C == checksquare(connectivities)
        @assert issymmetric(connectivities)
        return new{C,T}(sizes, connectivities)
    end
end

function StochasticBlockModel(
    total_size::Integer,
    nb_communities::Integer,
    connectivity_in::Real,
    connectivity_out::Real,
)
    sizes = fill(total_size ÷ nb_communities, nb_communities)
    connectivities = fill(connectivity_out, nb_communities, nb_communities)
    for c in 1:nb_communities
        connectivities[c, c] = connectivity_in
    end
    return StochasticBlockModel(sizes, connectivities)
end

const SBM = StochasticBlockModel

nb_vertices(sbm::SBM) = sum(sbm.sizes)
nb_communities(::SBM{C}) where {C} = C
community_size(sbm::SBM, c::Integer) = sbm.sizes[c]

function community_range(sbm::SBM, c::Integer)
    (; sizes) = sbm
    i = sum(view(sizes, 1:(c - 1)))
    j = i + sizes[c]
    return (i + 1):j
end

function community_of_vertex(sbm::SBM, v::Integer)
    for c in 1:nb_communities(sbm)
        if v in community_range(sbm, c)
            return c
        end
    end
    return 0
end

function Random.rand(rng::AbstractRNG, sbm::SBM{C}) where {C}
    (; sizes, connectivities) = sbm
    B = Matrix{SparseMatrixCSC{Bool,Int}}(undef, C, C)
    for c1 in 1:C, c2 in 1:C
        if c1 <= c2
            B[c1, c2] = sprand(rng, Bool, sizes[c1], sizes[c2], connectivities[c1, c2])
        else
            B[c1, c2] = spzeros(Bool, sizes[c1], sizes[c2])
        end
    end
    A = unnest_matrix_of_blocks(B)
    A .-= Diagonal(A)
    return sparse(Symmetric(A, :U))
end

struct ContextualStochasticBlockModel{C,T,F}
    sbm::SBM{C,T}
    features::Vector{F}

    function ContextualStochasticBlockModel(
        sbm::SBM{C,T}, features::Vector{F}
    ) where {C,T,F}
        @assert C == length(features)
        return new{C,T,F}(sbm, features)
    end
end

const CSBM = ContextualStochasticBlockModel
