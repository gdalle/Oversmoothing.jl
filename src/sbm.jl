function unnest_matrix_of_blocks(matrix_of_blocks)
    return mapreduce(Base.splat(hcat), vcat, eachrow(matrix_of_blocks))
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

function StochasticBlockModel(N::Integer, C::Integer, p_in::Real, p_out::Real)
    @assert N % C == 0
    sizes = fill(N ÷ C, C)
    connectivities = fill(p_out, C, C)
    for c in 1:C
        connectivities[c, c] = p_in
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

communities(sbm::SBM) = community_of_vertex.(Ref(sbm), 1:nb_vertices(sbm))

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
    A::SparseMatrixCSC{Bool,Int} = unnest_matrix_of_blocks(B)
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

function Random.rand(rng::AbstractRNG, csbm::CSBM)
    (; sbm, features) = csbm
    A = rand(rng, csbm.sbm)
    X = stack(1:nb_vertices(sbm); dims=1) do v
        c = community_of_vertex(sbm, v)
        rand(rng, features[c])
    end
    return A, X
end

feature_dimension(csbm::CSBM) = length(csbm.features[1])

## Specific models

function LinearCSBM1d(; N::Integer, C::Integer, p_in::Real, p_out::Real, σ::Real)
    sbm = SBM(N, C, p_in, p_out)
    μ = float.(1:C)
    Σ = fill(σ^2, C)
    features = UnivariateNormal.(μ, Σ)
    return CSBM(sbm, features)
end

function CircularCSBM2d(; N::Integer, C::Integer, p_in::Real, p_out::Real, σ::Real)
    sbm = SBM(N, C, p_in, p_out)
    r = inv(2 * sinpi(1 / C))
    μ = [r .* [cospi(2(c - 1) / C), sinpi(2(c - 1) / C)] for c in 1:C]
    @assert norm(μ[2] - μ[1]) ≈ 1
    Σ = [Diagonal([σ^2, σ^2]) for c in 1:C]
    features = BivariateNormal.(μ, Σ)
    return CSBM(sbm, features)
end
