using CairoMakie
using DensityInterface
using Oversmoothing

g1 = MultivariateGaussian([-3.0, -3.0], [1.0 0.1; 0.1 1.0])
g2 = MultivariateGaussian([2.0, 2.0], [1.0 -0.1; -0.1 1.0])
mix = Mixture([g1, g2], [0.3, 0.7])
xs = -5:0.1:5
ys = -5:0.1:5
zs1 = [logdensityof(mix, [x, y]) for x in xs, y in ys]

fig = Figure()
ax = Axis(fig[1, 1]; aspect=1)
contour!(ax, xs, ys, zs1; label="cluster 1", levels=10)
fig
