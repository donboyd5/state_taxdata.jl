# https://www.syncfusion.com/blogs/post/15-must-have-visual-studio-extensions-for-developers.aspx
# https://fluxml.ai/Flux.jl/stable/training/optimisers/
# https://julianlsolvers.github.io/Optim.jl/stable/#user/minimization/#_top

# %% example
using Flux

W = rand(2, 5)
b = rand(2)

predict(x) = (W * x) .+ b

x1 = [1, 2, 3, 4, 5]
W * x1
W * x1 .+ b
# https://julialang.org/blog/2017/01/moredots
# the caller indicates, by adding dots, which function calls and operators are intended to be applied elementwise (specifically, as broadcast calls)

predict(x1)

loss(x, y) = sum((predict(x) .- y).^2)

x, y = rand(5), rand(2) # Dummy data
l = loss(x, y) # ~ 3

θ = Flux.params(W, b)
grads = gradient(() -> loss(x, y), θ)

# We want to update each parameter, using the gradient, in order to improve (reduce) the loss. Here's one way to do that:

using Flux.Optimise: update!

η = 0.1 # Learning Rate
for p in (W, b)
  update!(p, η * grads[p])
end

# Running this will alter the parameters W and b and our loss should go down. Flux provides a more general way to do optimiser updates like this.

opt = Descent(0.1) # Gradient descent with learning rate 0.1

for p in (W, b)
  update!(opt, p, grads[p])
end


# %% another test
using Zygote

h(x, y) = 3x^2 + 2x + 1 + y*x - y
gradient(h, 3.0, 5.0)

# create an operator for grad of a function
D(f) = x-> gradient(f, x)[1]  # returns first in tuple

D_sin = D(sin)
sin(4.0)
D_sin(4.0)

# try with h above -- get first element of gradient vector
D2(f) = (x, y)-> gradient(f, x, y)[1]
D2h = D2(h)
D2h(3.0, 5.0)

# get all elements of the gradient vector
D2(f) = (x, y)-> gradient(f, x, y)
D2h = D2(h)
D2h(3.0, 5.0)

# can we make it more general
D2(f) = x-> gradient(f, x)
D2h = D2(h)
D2h(3.0, 5.0) # does not work
D2h((3.0, 5.0)) # does not work

# %% optim
using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

result = optimize(x-> x^2, -2.0, 1.0)

f(x) = x^2
result = optimize(x-> f(x), -2.0, 1.0)

x0 = 10
optimize(f, x0, Brent(); autodiff = :forward)

optimize(f, [10], LBFGS(); autodiff = :forward)


f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
x0 = [0.0, 0.0]
optimize(f, x0)
optimize(f, x0, NelderMead())
optimize(f, x0, LBFGS())
optimize(f, x0, LBFGS(); autodiff = :forward)

function g!(G, x)
  # gradient
  G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
  G[2] = 200.0 * (x[2] - x[1]^2)
  end

optimize(f, g!, x0, LBFGS())

function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

result = optimize(f, g!, h!, x0)

typeof(result)
# Optim.MultivariateOptimizationResults
fieldnames(result)
dump(result)
result.method
# result.method.Newton.alpha
result.initial_x
result.minimizer
minimizer(result)
summary(result)

f(result.minimizer)
f([1.0, 1.0])


f(x) = exp(x[1] + 7*x[2])
x0 = [1.0, 1.0]
f(x0)
rdef = optimize(f, x0; autodiff = :forward) # Nelder-Mead, success
rbg = optimize(f, x0, BFGS(); autodiff = :forward)
rlbg = optimize(f, x0, LBFGS(); autodiff = :forward)
rnewt = optimize(f, x0, Newton(); autodiff = :forward)

minimizer(rdef)
minimizer(rbg)
minimizer(rlbg)
minimizer(rnewt)
minimum(rdef)
minimum(rbg)
minimum(rlbg)
minimum(rnewt)
f(rdef.minimizer)
f(rbg.minimizer)

# %% reverse differentiation
# https://github.com/JuliaDiff/ReverseDiff.jl
# https://juliadiff.org/ReverseDiff.jl/api/
# https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl

using ReverseDiff
