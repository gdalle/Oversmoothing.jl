struct ErdosRenyi{T<:Real} <: AbstractRandomGraph
    N::Int
    q::T
end

const ER = ErdosRenyi

Base.show(io::IO, er::ER) = print(io, "ErdosRenyi($(er.N), $(er.q))")

nb_vertices(er::ER) = er.N
nb_communities(er::ER) = 1
community_range(er::ER, c::Integer) = 1:(er.N)

function Random.rand(rng::AbstractRNG, er::ER)
    A = sprand(rng, Bool, er.N, er.N, er.q)
    A .-= Diagonal(A)
    return Symmetric(A)
end
