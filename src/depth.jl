function error_by_depth(
    rng::AbstractRNG, csbm::CSBM; nb_layers, nb_trajectories, nb_graphs, nb_samples
)
    error_trajectories = random_walk_error_trajectories(
        rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples
    )
    return Particles.(Vector.(eachrow(error_trajectories)))
end

function optimal_depth(
    rng::AbstractRNG, csbm::CSBM; nb_layers, nb_trajectories, nb_graphs, nb_samples
)
    error_trajectories = random_walk_error_trajectories(
        rng, csbm; nb_layers, nb_trajectories, nb_graphs, nb_samples
    )
    return Particles(argmin.(eachcol(error_trajectories)))
end
