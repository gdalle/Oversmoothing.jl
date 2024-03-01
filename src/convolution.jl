
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
    mul!(scratch, A, H)
    H .+= scratch
    mul!(scratch, inv(D + I), H)
    return H .= scratch
end
