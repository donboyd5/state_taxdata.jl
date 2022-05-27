using LinearAlgebra
using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

# comment blocks: ctrl-k ctrl-c
# uncomment: clrl-k ctrl-u

# %% start by creating julia counterparts to selected python functions
# https://docs.julialang.org/en/v1/manual/functions/

# good, this takes a state-shares matrix and applies it to national weights to
# get each household's state weights

state_weights(wh, stateshares)

function state_weights(beta, wh, xmat)
    # beta: coefficients, s x k
    # wh: weight for each household, h values
    # xmat: hh characteristics, h x k
    # geotargets: s x k
    # betax = beta.dot(xmat.T)

    betaxp = beta * xmat'  # (s x k) * (k x h) = s x h

    # adjust betax to make exponentiation more stable numerically
    # subtract column-specific constant (the max) from each column of betax
    # const = betax.max(axis=0)
    betaxpmax = maximum(betaxp, dims=1) # max of each col of betaxp: 1 x h
    # betax = jnp.subtract(betax, const)
    betaxpadj = betaxp .- betaxpmax # (s x h) - (1 x h) = (s x h)
    ebetaxpadj = exp.(betaxpadj) # s x h
    # logdiffs = betax - jnp.log(ebetax.sum(axis=0))
    colsums = sum(ebetaxpadj, dims=1)  # 1 x h
    logdiffs = betaxpadj .- log.(colsums) # (s x h) - (1 x h) = (s x h)
    shares = exp.(logdiffs) # (s x h)
    # whs = jnp.multiply(wh, shares).T
    whs = (wh .* shares)' # (h x 1) x (h x s) = untransposed
    whs
end

whs = state_weights(beta_opt, wh, xmat)
sum(whs, dims=1)
sum(whs, dims=2)

sum(whs, dims=2) .- wh'

whs_opt

whs .- whs_opt

targets_opt = whs' * xmat
targets_opt - geotargets


targets_opt = whs_opt' * xmat
targets_opt - geotargets

sum(whs, dims=2) # check
wh

a = [-1.	3.
    2.	-2.
    0.	4.
    -3.	5.]
maximum(a, dims=1)

b = [6.	-1.	7.	8.
     9.	10.	1.	2.]

exp.(b)

ab = a * b
abmax = maximum(ab, dims=1)
abadj = ab .- abmax
eabadj = exp(abadj)
csums = sum(eabadj, dims=1)
abs.(csums)
log.(abs.(csums))

log.([1.0 2.0 3.0])

log(1)
exp(1)
log(exp(1))

exp(ab)

ab .- [1, 2, 3, 4]
state_weights(a, 1, b', 2)


# 21	31	-4	-2
# -6	-22	12	12
# 36	40	4	8
# 27	53	-16	-14


m =  [1 2 3 4;
      4 1 6 5;
      7 8 1 9]
m
sum(m, dims = 2)  # row sums
sum(m, dims=1) # col sums
stateshares = m ./ sum(m, dims=2) # note the ., which broadcasts
# colshares = m ./ sum(m, dims=1) # 3 people, 4
sum(stateshares, dims=2)

wh = [100, 200, 300]  # national weights-household for each of 3 households
wh = [100 200 300]# matrix
whs = wh .* stateshares
sum(whs, dims=2) # check




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


xmat



function targets_diff(beta, wh, xmat, geotargets)

end



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
