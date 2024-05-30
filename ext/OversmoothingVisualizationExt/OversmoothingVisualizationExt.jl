module OversmoothingVisualizationExt

using Base.Threads: @threads
using DensityInterface
using Distributions
using Makie
using OhMyThreads
using Oversmoothing
using Random: AbstractRNG

include("plot_1d.jl")
include("plot_2d.jl")
include("errors.jl")

end
