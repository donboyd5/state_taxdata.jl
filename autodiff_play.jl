# https://discourse.julialang.org/t/state-of-automatic-differentiation-in-julia/43083/3
# https://fluxml.ai/Flux.jl/stable/
# https://github.com/FluxML/model-zoo/
# https://github.com/FluxML/Flux.jl
# https://github.com/baggepinnen/FluxOptTools.jl

using Flux
using Zygote
using Optim
using FluxOptTools
using Statistics


m      = Chain(Dense(1,3,tanh) , Dense(3,1))
x      = LinRange(-pi,pi,100)'
y      = sin.(x)
loss() = mean(abs2, m(x) .- y)
Zygote.refresh()
pars   = Flux.params(m)
lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
res = Optim.optimize(Optim.only_fg!(fg!), p0, Optim.Options(iterations=1000, store_trace=true))

using Plots
plot(loss, pars, l=0.1, npoints=50, seriestype=:contour)


