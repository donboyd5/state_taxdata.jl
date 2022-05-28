using LinearAlgebra
using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

# comment blocks: ctrl-k ctrl-c
# uncomment: clrl-k ctrl-u

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

ibeta = zeros(size(beta_opt))

function f(beta)
    psf.objfn(beta, wh, xmat, geotargets)
end

# methods that do not require a gradient
res1 = optimize(f, ibeta, NelderMead())
res2 = optimize(f, ibeta, SimulatedAnnealing())  # max iterations
# methods that require a gradient
res3 = optimize(f, ibeta, BFGS(); autodiff = :forward)
res4 = optimize(f, ibeta, LBFGS(); autodiff = :forward)
res5 = optimize(f, ibeta, ConjugateGradient(); autodiff = :forward)
res6 = optimize(f, ibeta, GradientDescent(); autodiff = :forward)
res7 = optimize(f, ibeta, MomentumGradientDescent(); autodiff = :forward)
res8 = optimize(f, ibeta, AcceleratedGradientDescent(); autodiff = :forward)


res = res8
dump(res)
beta_o = minimizer(res)
whs_o = psf.geo_weights(beta_o, wh, xmat)
sum(whs_o, dims=2) .- wh'
geotarg_o = psf.geo_targets(whs_o, xmat)
geotargets
psf.targ_pdiffs(geotarg_o, geotargets)
psf.sspd(geotarg_o, geotargets)

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
