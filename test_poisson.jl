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

include("jl/rproblem.jl")

# %% start by creating julia counterparts to selected python functions
# https://docs.julialang.org/en/v1/manual/functions/

# good, this takes a state-shares matrix and applies it to national weights to
# get each household's state weights

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

function f(beta)
    # beta = reshape(beta, size(geotargets))
    psf.objfn(beta, wh, xmat, geotargets)
end

# methods that do not require a gradient
res1 = optimize(f, ibeta, NelderMead())
res2 = optimize(f, ibeta, SimulatedAnnealing())  # max iterations
# methods that require a gradient
res3 = optimize(f, ibeta, BFGS(); autodiff = :forward)

# two ways to do it
res4 = optimize(f, ibeta, LBFGS(); autodiff = :forward)
res4a = optimize(beta -> psf.objfn(beta, wh, xmat, geotargets), ibeta, LBFGS(); autodiff = :forward)

res5 = optimize(f, ibeta, ConjugateGradient(); autodiff = :forward)
res6 = optimize(f, ibeta, GradientDescent(); autodiff = :forward)
res7 = optimize(f, ibeta, MomentumGradientDescent(); autodiff = :forward)
res8 = optimize(f, ibeta, AcceleratedGradientDescent(); autodiff = :forward)

# hessian required
# https://julianlsolvers.github.io/Optim.jl/stable
# https://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/autodiff/
# https://github.com/JuliaNLSolvers/Optim.jl/tree/v0.4.5
# td = TwiceDifferentiable(f, ibeta; autodiff = :forward)
# res9 = optimize(td, vec(ibeta), Newton())
# Newton requires vector input
res9 = optimize(f, ibeta, Newton(); autodiff = :forward)
res10 = optimize(f, ibeta, NewtonTrustRegion(); autodiff = :forward)

# bvec = vec(beta_o)
beta_o
reshape(bvec, (3, 2))
reshape(beta_o, (3, 2))

optimize(f, ibeta, Newton(); autodiff = :forward)

res = res9
dump(res)
beta_o = reshape(minimizer(res), size(geotargets))
whs_o = psf.geo_weights(beta_o, wh, xmat)
sum(whs_o, dims=2) .- wh'
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


# %% Zygote
# https://fluxml.ai/Zygote.jl/latest/
# https://avik-pal.github.io/RayTracer.jl/dev/getting_started/optim_compatibility/
using Zygote
Zygote.gradient(x -> 3x^2 + 2x + 1, 5)

function f1(x)
    3x^2 + 2x + 1
end

g = Zygote.gradient(f1)

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
