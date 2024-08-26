function accuracy_zeroth_layer(csbm::CSBM; rtol)
    (; sbm, features) = csbm
    (; sizes) = sbm
    densities0 = features
    mix0 = Mixture(densities0, sizes ./ sum(sizes))
    return accuracy_quadrature(mix0; rtol)
end

function accuracy_first_layer(csbm::CSBM; max_neighbors, rtol)
    (; sbm) = csbm
    (; sizes) = sbm
    densities1 = first_layer_densities(csbm; max_neighbors)
    mix1 = Mixture(densities1, sizes ./ sum(sizes))
    return accuracy_quadrature(mix1; rtol)
end

function accuracy_by_depth(
    rng::AbstractRNG,
    csbm::CSBM,
    ::Val{method}=Val(:randomwalk);
    nb_layers,
    nb_trajectories,
    kwargs...,
) where {method}
    accuracy_trajectories = if method == :randomwalk
        random_walk_accuracy_trajectories(rng, csbm; nb_layers, nb_trajectories, kwargs...)
    elseif method == :logisticregression
        logistic_regression_accuracy_trajectories(
            rng, csbm; nb_layers, nb_trajectories, kwargs...
        )
    end
    return MonteCarloValue.(Vector.(eachrow(accuracy_trajectories)))
end
