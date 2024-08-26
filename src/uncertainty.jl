struct IntervalValue{T}
    val::T
    error::T
end

struct MonteCarloValue{T}
    samples::Vector{T}
end

value(v::Real) = v
value(iv::IntervalValue) = iv.val
value(mv::MonteCarloValue) = mean(mv.samples)

uncertainty(::T) where {T<:Real} = zero(T)
uncertainty(iv::IntervalValue) = iv.error
uncertainty(mv::MonteCarloValue) = std(mv.samples)
