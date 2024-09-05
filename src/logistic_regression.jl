function logistic_regression_accuracy_trajectories(
    rng::AbstractRNG, csbm::CSBM; nb_layers, nb_trajectories, nb_graphs
)
    T = nb_trajectories
    C = nb_communities(csbm.sbm)
    L = nb_layers
    D = feature_dimension(csbm)

    accuracy_trajectories = fill(NaN, L + 1, T)

    for t in 1:T
        embeddings_train = empirical_embeddings(rng, csbm; nb_layers, nb_graphs)
        embeddings_test = empirical_embeddings(rng, csbm; nb_layers, nb_graphs)

        targets_train = [
            fill(c, size(embeddings_train[l + 1, c], 1)) for l in 0:L, c in 1:C
        ]
        targets_test = [fill(c, size(embeddings_test[l + 1, c], 1)) for l in 0:L, c in 1:C]

        xs_train = [reduce(vcat, embeddings_train[l + 1, :]) for l in 0:L]
        xs_test = [reduce(vcat, embeddings_test[l + 1, :]) for l in 0:L]

        ys_train = [reduce(vcat, targets_train[l + 1, :]) for l in 0:L]
        ys_test = [reduce(vcat, targets_test[l + 1, :]) for l in 0:L]

        for l in 0:L
            x_train, y_train = xs_train[l + 1], ys_train[l + 1]
            x_test, y_test = xs_test[l + 1], ys_test[l + 1]
            model = MultinomialRegression(0.0; fit_intercept=false)
            W_vec = MLJLinearModels.fit(model, x_train, y_train)
            W = reshape(W_vec, D, C)
            y_test_pred = softmax(x_test * W; dims=2)
            accuracy = mean(getindex.(argmax(y_test_pred; dims=2), 2) .== y_test)
            accuracy_trajectories[l + 1, t] = accuracy
        end
    end

    return accuracy_trajectories
end
