abstract type AbstractRandomGraph end

community_size(graph::AbstractRandomGraph, c::Integer) = length(community_range(graph, c))

function community_of_vertex(graph::AbstractRandomGraph, v::Integer)
    return first(c for c in 1:nb_communities(graph) if v in community_range(graph, c))
end
