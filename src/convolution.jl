
abstract type AbstractConvolution end

struct NeighborhoodSum <: AbstractConvolution end
struct NeighborhoodAverage <: AbstractConvolution end

function apply!(
    H::AbstractVecOrMat,
    scratch::AbstractVecOrMat,
    A::AbstractMatrix,
    D::AbstractMatrix,
    ::NeighborhoodSum,
)
    mul!(scratch, A, H)
    return H .+= scratch
end

function apply!(
    H::AbstractVecOrMat,
    scratch::AbstractVecOrMat,
    A::AbstractMatrix,
    D::AbstractMatrix,
    ::NeighborhoodAverage,
)
    apply!(H, scratch, A, D, NeighborhoodSum())
    mul!(scratch, inv(D + I), H)
    return H .= scratch
end
