function random_walk_errors(
    rng::AbstractRNG, csbm::CSBM; nb_trajectories=1, nb_layers=1, nb_graphs=1
)
    (; sbm) = csbm
    (; sizes) = sbm
    error_trajectories = mapreduce(hcat, 1:nb_trajectories) do _
        mixtures = random_walk_mixtures(rng, csbm; nb_layers, nb_graphs)
        to_classify = [
            Mixture(mixtures[l + 1, :], sizes ./ sum(sizes)) for l in 0:nb_layers
        ]
        error_quadrature.(to_classify)
    end
    return error_trajectories
end

function best_depth(
    rng::AbstractRNG, csbm::CSBM; nb_trajectories=5, nb_layers=5, nb_graphs=5, kwargs...
)
    error_trajectories = random_walk_errors(
        rng, csbm; nb_trajectories, nb_layers, nb_graphs
    )
    return Particles(argmin.(eachcol(error_trajectories)) .- 1)
end
