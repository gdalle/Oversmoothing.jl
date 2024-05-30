module OversmoothingVisualizationExt

using Base.Threads: @threads
using Random: AbstractRNG

using DensityInterface
using Distributions
using Makie
using OhMyThreads
using Oversmoothing
using Oversmoothing: community_size, nb_communities, split_by_community

include("plot_1d.jl")
# include("plot_2d.jl")
# include("errors.jl")

end
