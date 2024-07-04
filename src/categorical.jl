struct Categorical{T<:Real}
    probs::Vector{T}
end

StatsBase.entropy(dist::Categorical) = -sum(dist.probs .* log.(dist.probs))
