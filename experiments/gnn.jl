using Pkg
Pkg.activate(@__DIR__)

using GraphNeuralNetworks, Zygote, Flux, Flux.Losses
using LinearAlgebra
using Oversmoothing
using Random
using Statistics
using SparseArrays

rng = Random.default_rng()

C = 3
D = 4
N = 100C

sbm = SBM(N, C, 0.03, 0.01)
features = [MultivariateNormal(randn(D), float.(I(D))) for c in 1:C]
csbm = CSBM(sbm, features)

all_graphs = map(1:100) do _
    A, X = rand(rng, csbm)
    y = community_of_vertex.(Ref(sbm), 1:N)
    g = GNNGraph(
        A;
        graph_type=:sparse,
        ndata=(; x=transpose(Float32.(X)), y=Flux.onehotbatch(y, 1:C)),
    )
end

train_graphs = all_graphs[1:5]
test_graphs = all_graphs[6:10]

loss(model, g::GNNGraph) = crossentropy(model(g, g.x), g.y)
loss(model, gs::Vector{<:GNNGraph}) = mean(loss(model, g) for g in gs)

model = GNNChain(GCNConv(D => D, relu), Dense(D => C), softmax)

g = first(all_graphs)
model(g, g.x)
sum(model(g, g.x); dims=1)

opt = Flux.setup(Adam(1.0f-3), model);

@profview for epoch in 1:100
    for g in train_graphs
        grad = gradient(model -> loss(model, g), model)
        Flux.update!(opt, model, grad[1])
    end

    @info (;
        epoch, train_loss=loss(model, train_graphs), test_loss=loss(model, test_graphs)
    )
end