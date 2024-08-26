struct IntervalValue{T}
    val::T
    error::T
end

struct MonteCarloValue{T}
    samples::Vector{T}
end

value(iv::IntervalValue) = iv.val
value(mv::MonteCarloValue) = mean(mv.samples)
