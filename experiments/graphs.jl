using LinearAlgebra
using Oversmoothing
using Random
using StableRNGs

rng = StableRNG(63)

er = ER(100, 0.2)
rand(rng, er)

contextual = Contextual(
    SBM([4, 6], [0.1 0.01; 0.02 0.2]),
    [Gaussian(ones(2), Diagonal(ones(2))), Gaussian(zeros(2), Diagonal(ones(2)))],
)
(; A, X) = rand(rng, contextual)
