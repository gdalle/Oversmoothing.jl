struct StochasticBlockModel{T<:Real} <: AbstractRandomGraph
    S::Vector{Int}
    Q::Matrix{T}

    function StochasticBlockModel(S::AbstractVector{Int}, Q::AbstractMatrix{T}) where {T}
        C = length(S)
        @assert C == checksquare(Q)
        @assert issymmetric(Q)
        return new{T}(S, Q)
    end
end

const SBM = StochasticBlockModel

Base.show(io::IO, sbm::SBM) = print(io, "SBM($(sbm.S), $(sbm.Q))")

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
    if C == 1
        N = nb_vertices(sbm)
        A = sprand(rng, Bool, N, N, Q[1, 1])
        A .-= Diagonal(A)
        return Symmetric(A), :U
    elseif C == 2
        N1, N2 = S
        A11 = sprand(rng, Bool, N1, N1, Q[1, 1])
        A12 = sprand(rng, Bool, N1, N2, Q[1, 2])
        A21 = spzeros(Bool, N2, N1)
        A22 = sprand(rng, Bool, N2, N2, Q[2, 2])
        A = [
            A11 A12
            A21 A22
        ]
        A .-= Diagonal(A)
        return Symmetric(A, :U)
    elseif C == 3
        N1, N2, N3 = S
        A11 = sprand(rng, Bool, N1, N1, Q[1, 1])
        A12 = sprand(rng, Bool, N1, N2, Q[1, 2])
        A13 = sprand(rng, Bool, N1, N3, Q[1, 3])
        A21 = spzeros(Bool, N2, N1)
        A22 = sprand(rng, Bool, N2, N2, Q[2, 2])
        A23 = sprand(rng, Bool, N2, N3, Q[2, 3])
        A31 = spzeros(Bool, N3, N1)
        A32 = spzeros(Bool, N3, N2)
        A33 = sprand(rng, Bool, N3, N3, Q[3, 3])
        A = [
            A11 A12 A13
            A21 A22 A23
            A31 A32 A33
        ]
        A .-= Diagonal(A)
        return Symmetric(A, :U)
    else
        error("$C communities not supported")
    end
end
