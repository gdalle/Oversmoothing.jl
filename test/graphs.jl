using LinearAlgebra
using Oversmoothing
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

N = 1000
M = 2000
q = 0.05
Q = [0.05 0.01; 0.01 0.03]

B = bernoulli_matrix(rng, M, N, q);
@test eltype(B) == Bool
@test size(B) == (M, N)
@test mean(B) ≈ q rtol = 1e-1

er = ER(N, q)
A = rand(rng, er)
@test length(er) == N
@test community_size(er, 1) == N
@test get_community.(Ref(er), 1:N) == fill(1, N)
@test A isa Symmetric
@test mean(A) ≈ q atol = 1e-1

sbm = SBM([N, 2N], Q)
A = rand(rng, sbm)
@test length(sbm) == 3N
@test community_size.(Ref(sbm), 1:2) == [N, 2N]
@test get_community.(Ref(sbm), 1:(3N)) == vcat(fill(1, N), fill(2, 2N))
@test A isa Symmetric
@test mean(view(A, 1:N, 1:N)) ≈ Q[1, 1] rtol = 1e-1
@test mean(view(A, 1:N, N:(3N))) ≈ Q[1, 2] rtol = 1e-1
@test mean(view(A, N:(3N), 1:N)) ≈ Q[2, 1] rtol = 1e-1
@test mean(view(A, N:(3N), N:(3N))) ≈ Q[2, 2] rtol = 1e-1
