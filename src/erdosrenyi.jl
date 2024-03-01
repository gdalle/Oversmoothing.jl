
@kwdef struct ErdosRenyi{T<:Real} <: AbstractRandomGraph
    N::Int
    q::T
end

const ER = ErdosRenyi

Base.length(er::ER) = er.N
nb_communities(er::ER) = 1
get_community(er::ER, v::Integer) = 1
community_size(er::ER, c::Integer) = length(er)

function Random.rand(rng::AbstractRNG, er::ER)
    A = bernoulli_matrix(rng, er.N, er.N, er.q)
    A .-= Diagonal(A)
    return Symmetric(A)
end

function bernoulli_matrix(rng::AbstractRNG, m::Integer, n::Integer, q::Real)
    selection = randsubseq(rng, 1:(m * n), q)
    is = mod1.(selection, m)
    js = fld1.(selection, m)
    vs = ones(Bool, length(selection))
    return sparse(is, js, vs, m, n)
end
