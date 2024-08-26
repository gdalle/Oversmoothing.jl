struct RandomWalkConv <: GNNLayer
    iterations::Int
end

Flux.@layer RandomWalkConv

function (model::RandomWalkConv)(g::GNNGraph, x::AbstractMatrix)
    A = GraphNeuralNetworks.adjacency_matrix(g)
    W = transpose(random_walk(A))
    y = x
    for _ in 1:(model.iterations)
        y = y * W
    end
    return y
end

onecold(x) = getindex.(argmax(x; dims=1), 1)

loss(model, g::GNNGraph) = crossentropy(model(g, g.x), g.y)
loss(model, gs) = mean(loss(model, g) for g in gs)

function gnn_accuracy_trajectories(
    rng::AbstractRNG,
    csbm::CSBM;
    nb_layers,
    nb_trajectories,
    nb_train_graphs,
    nb_test_graphs,
    nb_epochs,
    learning_rate,
    batch_size,
)
    (; sbm,) = csbm
    N = nb_vertices(sbm)
    C = nb_communities(sbm)
    D = feature_dimension(csbm)
    accuracy_trajectories = fill(NaN, nb_layers + 1, nb_trajectories)
    for t in 1:nb_trajectories
        all_graphs = map(1:(nb_train_graphs + nb_test_graphs)) do _
            A, X = rand(rng, csbm)
            y = communities(csbm.sbm)
            GNNGraph(
                A;
                graph_type=:sparse,
                ndata=(; x=Matrix(transpose(Float32.(X))), y=onehotbatch(y, 1:C)),
            )
        end
        train_graphs = DataLoader(
            first(all_graphs, nb_train_graphs);
            batchsize=batch_size,
            shuffle=true,
            collate=true,
        )
        test_graphs = DataLoader(
            last(all_graphs, nb_test_graphs);
            batchsize=batch_size,
            shuffle=true,
            collate=true,
        )
        @progress "Trajectory $t / $nb_trajectories" for l in 0:nb_layers
            # model
            mlp = GNNChain(Dense(D => C; init=Flux.glorot_uniform(rng)), softmax)
            model = GNNChain(RandomWalkConv(l), mlp)
            opt = Flux.setup(Adam(learning_rate), model)

            # train
            for _ in 1:nb_epochs
                for g in train_graphs
                    grad = gradient(model -> loss(model, g), model)
                    Flux.update!(opt, model, grad[1])
                end
            end

            # eval
            accuracy = mean(
                mean(onecold(model(g, g.x)) .== onecold(g.y)) for g in test_graphs
            )
            accuracy_trajectories[l + 1, t] = accuracy
        end
    end
    return accuracy_trajectories
end
