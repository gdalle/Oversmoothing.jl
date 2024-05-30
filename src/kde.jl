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
    kl = sum(eachrow(X)) do row
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
    X_split_train::Vector{<:AbstractMatrix},
    X_split_test::Vector{<:AbstractMatrix};
    kwargs...,
)
    C = length(X_split_train)
    P = [density_estimator(X_split_train[c]; kwargs...) for c in 1:C]
    error_count = 0
    for c in 1:C
        for i in axes(X_split_test[c], 1)
            x = Tuple(view(X_split_test[c], i, :))
            likelihoods = [length(X_split_train[c]) * pdf(P[c], x...) for c in 1:C]
            c_max = argmax(likelihoods)
            if c_max != c
                error_count += 1
            end
        end
    end
    return error_count / sum(length, X_split_test)
end
