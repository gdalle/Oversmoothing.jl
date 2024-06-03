const MvNormalMixture = MixtureModel{Multivariate,Continuous,<:MvNormal}

function squared_mean(m)
    μ = mean(m)
    return μ * transpose(μ)
end

function mixture_cov(mix::MvNormalMixture)
    c, p = components(mix), probs(mix)
    second_moment = sum(p[i] * (_cov(c[i]) + squared_mean(c[i])) for i in eachindex(c, p))
    Σ = second_moment - squared_mean(mix)
    return Σ
end
