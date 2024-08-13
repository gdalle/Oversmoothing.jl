function error_zeroth_layer(csbm::CSBM; rtol)
    (; sbm, features) = csbm
    (; sizes) = sbm
    densities0 = features
    mix0 = Mixture(densities0, sizes ./ sum(sizes))
    error0 = error_quadrature(mix0; rtol)
    return error0
end

function error_first_layer(csbm::CSBM; max_neighbors, rtol)
    (; sbm, features) = csbm
    (; sizes) = sbm
    densities1 = first_layer_densities(csbm; max_neighbors)
    mix1 = Mixture(densities1, sizes ./ sum(sizes))
    error1 = error_quadrature(mix1; rtol)
    return error1
end

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
