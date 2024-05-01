@kwdef struct StochasticBlockModel{T<:Real} <: AbstractRandomGraph
    S::Vector{Int}
    Q::Matrix{T}
end

const SBM = StochasticBlockModel

nb_vertices(sbm::SBM) = sum(sbm.S)
nb_communities(sbm::SBM) = length(sbm.S)

function community_range(sbm::SBM, c::Integer)
    (; S) = sbm
    i = sum(view(S, 1:(c - 1)))
    j = i + S[c]
    return (i + 1):j
end

function Random.rand(rng::AbstractRNG, sbm::SBM)
    (; S, Q) = sbm
    C = nb_communities(sbm)
    A = BlockArray(undef_blocks, SparseMatrixCSC{Bool,Int64}, S, S)
    for c1 in 1:C, c2 in 1:C
        A_block = bernoulli_matrix(rng, S[c1], S[c2], Q[c1, c2])
        if c1 == c2
            A_block .-= Diagonal(A_block)
        end
        A[Block(c1, c2)] = A_block
    end
    return Symmetric(A)
end
