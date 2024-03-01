
degree_matrix(A::AbstractMatrix) = Diagonal(map(sum, eachrow(A)))

struct NeighborhoodSum{M<:AbstractMatrix} <: AbstractConvolution
    A::M
end

function apply!(H, scratch, conv::NeighborhoodSum)
    mul!(scratch, conv.A, H)
    scratch .+= H
    H .= scratch
    return H
end

struct NeighborhoodAverage{M<:AbstractMatrix,DM<:Diagonal} <: AbstractConvolution
    A::M
    inv_D_plus_I::DM
    function NeighborhoodAverage(A)
        D = degree_matrix(A)
        inv_D_plus_I = inv(D + I)
        return new{typeof(A),typeof(inv_D_plus_I)}(A, inv_D_plus_I)
    end
end

function apply!(H, scratch, conv::NeighborhoodAverage)
    mul!(scratch, conv.A, H)
    scratch .+= H
    H .= scratch
    mul!(scratch, conv.inv_D_plus_I, H)
    H .= scratch
    return H
end

function single_coeff(conv::AbstractConvolution, layers::Integer, u::Integer, v::Integer)
    h = zeros(size(conv.A, 2))
    h[v] = 1
    scratch = similar(h)
    for l in 1:layers
        apply!(h, scratch, conv)
    end
    return h[u]
end
