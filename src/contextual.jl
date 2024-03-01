struct Contextual{G<:AbstractRandomGraph,F<:AbstractMeasure}
    graph::G
    features::Vector{F}
end

Base.length(contextual::Contextual) = length(contextual.graph)
nb_communities(contextual::Contextual) = nb_communities(contextual.graph)
community_size(contextual::Contextual, c::Integer) = community_size(contextual.graph, c)
get_community(contextual::Contextual, v::Integer) = get_community(contextual.graph, v)

function Random.rand(rng::AbstractRNG, contextual::Contextual)
    (; graph, features) = contextual
    A = rand(rng, graph)
    X = map(1:length(graph)) do v
        c = get_community(graph, v)
        rand(rng, features[c])
    end
    return (; A, X)
end
