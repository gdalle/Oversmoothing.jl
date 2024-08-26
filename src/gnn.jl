struct RandomWalkConv <: GNNLayer end

Flux.@layer RandomWalkConv

function (model::RandomWalkConv)(g::GNNGraph, x)
    A = adjacency_matrix(g)
    W = random_walk(A)
    return transpose(W * transpose(x))
end

onecold(x) = getindex.(argmax(x; dims=1), 1)

loss(model, g::GNNGraph) = crossentropy(model(g, g.x), g.y)
loss(model, gs) = mean(classification_loss(model, g) for g in gs)

function gnn_error_trajectories(
    rng::AbstractRNG,
    csbm::CSBM;
    nb_layers,
    nb_trajectories,
    nb_graphs,
    nb_epochs,
    learning_rate,
)
    (; sbm,) = csbm
    N = nb_vertices(sbm)
    C = nb_communities(sbm)
    D = feature_dimension(csbm)
    error_trajectories = fill(NaN, nb_layers + 1, nb_trajectories)
    for t in 1:nb_trajectories
        all_graphs = map(1:(2nb_graphs)) do _
            A, X = rand(rng, csbm)
            y = communities(csbm.sbm)
            GNNGraph(
                A;
                graph_type=:sparse,
                ndata=(; x=transpose(Float32.(X)), y=onehotbatch(y, 1:C)),
            )
        end
        train_graphs = all_graphs[1:nb_graphs]
        test_graphs = all_graphs[(nb_graphs + 1):(2nb_graphs)]
        for l in 0:nb_layers
            @info "t=$t, l=$l"
            # model
            mlp = GNNChain(Dense(D => C, relu), softmax)
            model = GNNChain(fill(RandomWalkConv(), l)..., mlp)
            opt = Flux.setup(Adam(learning_rate), model)

            # train
            for epoch in 1:nb_epochs
                @info "$epoch"
                for g in train_graphs
                    grad = gradient(model -> loss(model, g), model)
                    Flux.update!(opt, model, grad[1])
                end
            end

            # eval
            error = mean(mean, onecold(model(g, g.x)) .!= onecold(g.y) for g in test_graphs)
            error_trajectories[l + 1, t] = error
        end
    end
    return error_trajectories
end
