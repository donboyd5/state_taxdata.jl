using LinearAlgebra
using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

# comment blocks: ctrl-k ctrl-c
# uncomment: clrl-k ctrl-u

# https://discourse.julialang.org/t/what-is-the-difference-between-zygote-vs-forwarddiff-and-reversediff/55789/3
# https://juliadiff.org/

# fix data viewer problem
# https://stackoverflow.com/questions/67698176/error-loading-webview-error-could-not-register-service-workers-typeerror-fai

# %% TODO:
#  reverse differentiation
#  make test problem
#  actual problems

# %% load local modules
include("jl/poisson_functions.jl")
import .poissonFunctions as psf

include("jl/make_test_problems.jl")
import .makeTestProblems as mtp

include("jl/rproblem.jl")

# %% start by creating julia counterparts to selected python functions
# https://docs.julialang.org/en/v1/manual/functions/

# good, this takes a state-shares matrix and applies it to national weights to
# get each household's state weights

function f(beta)
    # beta = reshape(beta, size(geotargets))
    psf.objfn(beta, wh, xmat, geotargets)
end

tp = mtp.mtp(10, 3, 2)
tp = mtp.mtp(100, 8, 4)
tp = mtp.mtp(1000, 12, 6)
tp = mtp.mtp(10000, 25, 12)
tp = mtp.mtp(50000, 50, 20)

xmat = tp[:xmat]
wh = tp[:wh]
whs = tp[:whs]
geotargets = tp[:geotargets]
targets = tp[:targets]

ibeta = zeros(length(geotargets))
# methods that do not require a gradient
res1 = optimize(f, ibeta, NelderMead(),
Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true))

res2 = optimize(f, ibeta, SimulatedAnnealing())  # max iterations
# methods that require a gradient

res3 = optimize(f, ibeta, BFGS(),
  Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
   autodiff = :forward) # gets slow with BIG problems

# two ways to do it
res4 = optimize(f, ibeta, LBFGS(),
Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
 autodiff = :forward)
res4a = optimize(beta -> psf.objfn(beta, wh, xmat, geotargets), ibeta, LBFGS(); autodiff = :forward)

res5 = optimize(f, ibeta, ConjugateGradient(),
Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
 autodiff = :forward)

 res6 = optimize(f, ibeta, GradientDescent(),
Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
 autodiff = :forward) # seems to become very slow as problem size increases

res7 = optimize(f, ibeta, MomentumGradientDescent(),
Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
 autodiff = :forward)

res8 = optimize(f, ibeta, AcceleratedGradientDescent(),
  Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
   autodiff = :forward)

# hessian required
# https://julianlsolvers.github.io/Optim.jl/stable
# https://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/autodiff/
# https://github.com/JuliaNLSolvers/Optim.jl/tree/v0.4.5
# td = TwiceDifferentiable(f, ibeta; autodiff = :forward)
# res9 = optimize(td, vec(ibeta), Newton())
# Newton requires vector input
res9 = optimize(f, ibeta, Newton(),
Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
 autodiff = :forward) # seems to get slow as problem gets large

res10 = optimize(f, ibeta, NewtonTrustRegion(),
  Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true);
   autodiff = :forward)

# bvec = vec(beta_o)
# beta_o
# reshape(bvec, (3, 2))
# reshape(beta_o, (3, 2))

# optimize(f, ibeta, Newton(); autodiff = :forward)

res = res7
dump(res)
beta_o = reshape(minimizer(res), size(geotargets))
whs_o = psf.geo_weights(beta_o, wh, xmat)
whdiffs = sum(whs_o, dims=2) .- wh
whdiffs ./ wh
geotarg_o = psf.geo_targets(whs_o, xmat)
geotargets
psf.targ_pdiffs(geotarg_o, geotargets)
psf.sspd(geotarg_o, geotargets)


# %% reverse differentiation
# https://github.com/JuliaDiff/ReverseDiff.jl/blob/master/examples/gradient.jl
# f(a, b) = sum(a' * b + a * b')
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile, DiffResults

# pre-record a GradientTape for `f` using inputs of shape 100x100 with Float64 elements
# const f_tape = GradientTape(f, (rand(100, 100), rand(100, 100)))
const f_tape = GradientTape(f, rand(100))

# compile `f_tape` into a more optimized representation
const compiled_f_tape = compile(f_tape)

# some inputs and work buffers to play around with
a, b = rand(100, 100), rand(100, 100)
inputs = (a, b)
results = (similar(a), similar(b))
all_results = map(DiffResults.GradientResult, results)
cfg = GradientConfig(inputs)

# https://www.youtube.com/watch?v=B5O3xBolDCc
# https://github.com/cpfiffer/julia-bootcamp-2022/blob/main/session-3/optimization-lecture.ipynb
using ReverseDiff
# ReverseDiff only works with array inputs
# f3(x::Vector{Float64}) = x^3
f3(x) = x^3
f3(4.0)
f3([4.0, 2.0])
gf3(x) = ReverseDiff.gradient(z -> f3(z[1]), x)[1]
gf3([4.0])
x = [4.0]
gf3(x)

# next does not work yet
gf3a = ReverseDiff.gradient(f3, x)
gf3a = ReverseDiff.gradient(f3, x[1])

# %% Zygote
# https://fluxml.ai/Zygote.jl/latest/
# https://avik-pal.github.io/RayTracer.jl/dev/getting_started/optim_compatibility/
using Zygote
Zygote.gradient(x -> 3x^2 + 2x + 1, 5)

function f1(x)
    3x^2 + 2x + 1
end

g = Zygote.gradient(f1, 5)
g
typeof(g)

Zygote.gradient(f1, x)

Zygote.gradient(f1)
gradient(model -> sum(model(x)), model)

# %% misc notes

wh

whs = psf.geo_weights(beta_opt, wh, xmat)
sum(whs, dims=1)
sum(whs, dims=2)

sum(whs, dims=2) .- wh'

calctargets = psf.geo_targets(whs, xmat)
geotargets

psf.sspd(calctargets, geotargets)

psf.objfn(beta_opt, wh, xmat, geotargets)

size(beta_opt)

ibeta = zeros(size(geotargets))
ibeta = vec(ibeta)
length(geotargets)

# %% end
stop
# def get_whs_logs(beta_object, wh, xmat, geotargets):
#     # note beta is an s x k matrix
#     # beta must be a matrix so if beta_object is a vector, reshape it
#     if beta_object.ndim == 1:
#         beta = beta_object.reshape(geotargets.shape)
#     elif beta_object.ndim == 2:
#         beta = beta_object

#     betax = beta.dot(xmat.T)
#     # adjust betax to make exponentiation more stable numerically
#     # subtract column-specific constant (the max) from each column of betax
#     const = betax.max(axis=0)
#     betax = jnp.subtract(betax, const)
#     ebetax = jnp.exp(betax)
#     # print(ebetax.min())
#     # print(np.log(ebetax))
#     logdiffs = betax - jnp.log(ebetax.sum(axis=0))
#     shares = jnp.exp(logdiffs)
#     whs = jnp.multiply(wh, shares).T
#     return whs


# %% selected old python functions

# def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
#     whs = get_whs_logs(beta_object, wh, xmat, geotargets)
#     geotargets_calc = jnp.dot(whs.T, xmat)
#     diffs = geotargets_calc - geotargets
#     diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

#     # return a matrix or vector, depending on the shape of beta_object
#     if beta_object.ndim == 1:
#         diffs = diffs.flatten()

#     # CAUTION: the return type is immutable and will not work with
#     # scipy's least_squares, hence the diff_copy version in the
#     # function jax_targets_diff_copy. I have not been able to incorporate
#     # the copy into this function successfully.
#     return diffs

# def get_whs_logs(beta_object, wh, xmat, geotargets):
#     # note beta is an s x k matrix
#     # beta must be a matrix so if beta_object is a vector, reshape it
#     if beta_object.ndim == 1:
#         beta = beta_object.reshape(geotargets.shape)
#     elif beta_object.ndim == 2:
#         beta = beta_object

#     betax = beta.dot(xmat.T)
#     # adjust betax to make exponentiation more stable numerically
#     # subtract column-specific constant (the max) from each column of betax
#     const = betax.max(axis=0)
#     betax = jnp.subtract(betax, const)
#     ebetax = jnp.exp(betax)
#     # print(ebetax.min())
#     # print(np.log(ebetax))
#     logdiffs = betax - jnp.log(ebetax.sum(axis=0))
#     shares = jnp.exp(logdiffs)
#     whs = jnp.multiply(wh, shares).T
#     return whs


# def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
#     whs = get_whs_logs(beta_object, wh, xmat, geotargets)
#     geotargets_calc = jnp.dot(whs.T, xmat)
#     diffs = geotargets_calc - geotargets
#     diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

#     # return a matrix or vector, depending on the shape of beta_object
#     if beta_object.ndim == 1:
#         diffs = diffs.flatten()

#     # CAUTION: the return type is immutable and will not work with
#     # scipy's least_squares, hence the diff_copy version in the
#     # function jax_targets_diff_copy. I have not been able to incorporate
#     # the copy into this function successfully.
#     return diffs


# def jax_sspd(beta_object, wh, xmat, geotargets, diff_weights):
#     diffs = jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights)
#     sspd = jnp.square(diffs).sum()
#     return sspd
