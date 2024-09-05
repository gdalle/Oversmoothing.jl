struct IntervalValue{T}
    val::T
    error::T
end

struct MonteCarloValue{T}
    samples::Vector{T}
end

value(v::Real) = v
value(mv::MonteCarloValue) = mean(mv.samples)

uncertainty(::T) where {T<:Real} = zero(T)
uncertainty(mv::MonteCarloValue) = std(mv.samples)
