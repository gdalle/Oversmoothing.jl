using LinearAlgebra
using Oversmoothing
using StableRNGs
using Statistics
using Test

rng = StableRNG(63)

B = [rand(4, 5) for i in 1:2, j in 1:3]
A = [B[1, 1] B[1, 2] B[1, 3]; B[2, 1] B[2, 2] B[2, 3]]
@test Oversmoothing.unnest_matrix_of_blocks(B) == A

@testset "2 communities" begin
    N = 1000
    Q = [0.05 0.01; 0.01 0.03]
    sbm = SBM([N, 2N], Q)
    A = @inferred rand(rng, sbm)
    W = @inferred Oversmoothing.random_walk(A)

    @test nb_vertices(sbm) == 3N
    @test community_size.(Ref(sbm), 1:2) == [N, 2N]
    @test community_of_vertex.(Ref(sbm), 1:(3N)) == vcat(fill(1, N), fill(2, 2N))
    @test issymmetric(A)
    @test all(iszero, Diagonal(A))
    @test mean(A[1:N, 1:N]) ≈ Q[1, 1] rtol = 1e-1
    @test mean(A[1:N, (N + 1):(3N)]) ≈ Q[1, 2] rtol = 1e-1
    @test mean(A[(N + 1):(3N), 1:N]) ≈ Q[2, 1] rtol = 1e-1
    @test mean(A[(N + 1):(3N), (N + 1):(3N)]) ≈ Q[2, 2] rtol = 1e-1
end

@testset "3 communities" begin
    N = 1000
    sbm = SBM(3N, 3, 0.03, 0.01)
    A = @inferred rand(rng, sbm)
    W = @inferred Oversmoothing.random_walk(A)

    @test nb_vertices(sbm) == 3N
    @test community_size.(Ref(sbm), 1:3) == [N, N, N]
    @test community_of_vertex.(Ref(sbm), 1:(3N)) == vcat(fill(1, N), fill(2, N), fill(3, N))
    @test issymmetric(A)
    @test all(iszero, Diagonal(A))
    @test mean(A[1:N, 1:N]) ≈ 0.03 rtol = 1e-1
    @test mean(A[1:N, (N + 1):(2N)]) ≈ 0.01 rtol = 1e-1
end
