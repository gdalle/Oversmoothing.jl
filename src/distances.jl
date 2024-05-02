function density_estimator(X::AbstractMatrix; kwargs...)
    if size(X, 2) == 1
        return kde(X[:, 1]; kwargs...)
    elseif size(X, 2) == 2
        return kde(X; kwargs...)
    end
end

function empirical_kl(X::AbstractMatrix, Y::AbstractMatrix; kwargs...)
    P = density_estimator(X; kwargs...)
    Q = density_estimator(Y; kwargs...)
    kl = 0.0
    for row in eachrow(X)
        x = Tuple(row)
        Px = pdf(P, x...)
        Qx = pdf(Q, x...)
        if !iszero(Px)
            kl += Px * log(Px / Qx)
        end
    end
    return kl
end

function misclassification_probability(
    rng::AbstractRNG, X_split::Vector{<:AbstractMatrix}; samples=100, kwargs...
)
    X = stack(X_split; dims=1)
    S = length.(X_split)
    P = [kde(X; kwargs...) for X in X_split]
    proba = 0.0
    for _ in 1:samples
        i = rand(1:size(X, 1))
        x = Tuple(view(X, i, :))
        likelihoods = [S[c] * pdf(P[c], x) for c in eachindex(P)]
        c_max = argmax(likelihoods)
    end
    return proba
end
