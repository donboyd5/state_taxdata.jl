# %% imports
import numpy as np
import scipy

import jax
import jax.numpy as jnp

# from jax import jvp, vjp

# # this next line is CRUCIAL or we will lose precision
from jax.config import config
config.update('jax_enable_x64', True)


# %% scaling functions
def scale_problem(xmat, geotargets, scale_goal):
    scale_factors = xmat.sum(axis=0) / scale_goal
    xmat = jnp.divide(xmat, scale_factors)
    geotargets = jnp.divide(geotargets, scale_factors)
    return xmat, geotargets, scale_factors


# %% weight calculation functions
def jax_get_diff_weights(geotargets, goal=100):
    # establish a weight for each target that, prior to application of any
    # other weights, will give each target equal priority
    goalmat = jnp.full(geotargets.shape, goal)
    # djb note there is no jnp.errstate so I use np.errstate
    # with np.errstate(divide='ignore'):  # turn off divide-by-zero warning
    diff_weights = jnp.where(geotargets != 0, goalmat / geotargets, 1)
    return diff_weights


def get_whs_logs(beta_object, wh, xmat, geotargets):
    # note beta is an s x k matrix
    # beta must be a matrix so if beta_object is a vector, reshape it
    if beta_object.ndim == 1:
        beta = beta_object.reshape(geotargets.shape)
    elif beta_object.ndim == 2:
        beta = beta_object

    betax = beta.dot(xmat.T)
    # adjust betax to make exponentiation more stable numerically
    # subtract column-specific constant (the max) from each column of betax
    const = betax.max(axis=0)
    betax = jnp.subtract(betax, const)
    ebetax = jnp.exp(betax)
    # print(ebetax.min())
    # print(np.log(ebetax))
    logdiffs = betax - jnp.log(ebetax.sum(axis=0))
    shares = jnp.exp(logdiffs)
    whs = jnp.multiply(wh, shares).T
    return whs


def jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights):
    whs = get_whs_logs(beta_object, wh, xmat, geotargets)
    geotargets_calc = jnp.dot(whs.T, xmat)
    diffs = geotargets_calc - geotargets
    diffs = jnp.divide(diffs, geotargets) * 100.0  # can't have zero geotargets

    # return a matrix or vector, depending on the shape of beta_object
    if beta_object.ndim == 1:
        diffs = diffs.flatten()

    # CAUTION: the return type is immutable and will not work with
    # scipy's least_squares, hence the diff_copy version in the
    # function jax_targets_diff_copy. I have not been able to incorporate
    # the copy into this function successfully.
    return diffs

def jax_targets_diff_copy(beta_object, wh, xmat, geotargets, diff_weights):
    diffs = jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights)
    # copy gives us a mutable type, which is needed for scipy least_squares
    diffs = np.copy(diffs)
    return diffs

def jax_sspd(beta_object, wh, xmat, geotargets, diff_weights):
    diffs = jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights)
    sspd = jnp.square(diffs).sum()
    return sspd


# %% build jacobian from vectors functions

# define the different functions that can be used to construct the jacobian
# these are alternative to each other - we'll only use one
def jac_vjp(g, wh, xmat, geotargets, dw):
    # build jacobian row by row to conserve memory use
    f = lambda x: g(x, wh, xmat, geotargets, dw)
    def jacfun(x, wh, xmat, geotargets, dw):
        y, _vjp = jax.vjp(f, x)
        Jt, = jax.vmap(_vjp, in_axes=0)(jnp.eye(len(y)))
        return jnp.transpose(Jt)
    return jacfun

def jac_jvp(g, wh, xmat, geotargets, dw):
    # build jacobian column by column to conserve memory use
    f = lambda x: g(x, wh, xmat, geotargets, dw)
    def jacfun(x, wh, xmat, geotargets, dw):
        _jvp = lambda s: jax.jvp(f, (x,), (s,))[1]
        # gc.collect()
        Jt = jax.vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun


def jvp_linop(beta, wh, xmat, geotargets, dw):
    # linear operator approach
    # CAUTION: This does NOT work well because scipy least_squares does not allow the option x_scale='jac' when using a linear operator
    # This is fast and COULD be very good if a good scaling vector is developed but without that it iterates quickly but reduces
    # cost very slowly.
    l_diffs = lambda beta: jax_targets_diff(beta, wh, xmat, geotargets, dw)
    # l_diffs = jax.jit(l_diffs)  # jit is slower
    # l_jvp = lambda diffs: jax.jvp(l_diffs, (beta,), (diffs,))[1]  # need to reshape
    l_vjp = lambda diffs: jax.vjp(l_diffs, beta)[1](diffs)

    def f_jvp(diffs):
        diffs = diffs.reshape(diffs.size)
        return jax.jvp(l_diffs, (beta,), (diffs,))[1]
    # f_jvp = jax.jit(f_jvp)  # jit is slower

    linop = scipy.sparse.linalg.LinearOperator((beta.size, beta.size),
        matvec=f_jvp, rmatvec=l_vjp)
    return linop




# %% linear operator functions
# In concept, we need the inverse of the jacobian to compute a Newton step.
# However, we are avoiding creating the jacobian because we don't want to use
# that much memory or do such extensive calculations.
#
# To calculate the step, we need to solve the system:
#   Ax = b
#
# where:
#   A is the jacobian
#   b is the vector of differences from targets, evaluated at current bvec
#   x, to be solved for, is the next step we will take.
#
# If we were to create the jacobian (A in this example), we could invert it
# (if invertible). However, we avoid creating the jacobian through use of
# the l_jvp and l_vjp functions. Furthermore, this jacobian often seems
# extremely ill-conditioned, and so we don't use one of the methods that
# can solve for x iteratively, such as conjugate gradient
# (jax.scipy.sparse.linalg.cg) or gmres (scipy.sparse.linalg.gmres). I have
# tried these methods and they generally either fail because the jacobian is
# so ill-conditioned, or cannot be used because it is not positive semidefinite.
#
# Thus, as an alternative to solving for x directly (by constructing a large,
# difficult to calculate, ill-conditioned jacobian) or solving for x iteratively
# (by using an interative method such as cg or gmres), instead we solve for x
# approximately using least squares.
#
# Furthermore, we can solve approximately for x this way without creating the
# jacobian (matrix A in the system notation) by using the two vector-product
# functions l_jvp and l_vjp, and wrapping them in a linear operator. That's
# what this next line does. This lets us solve for approximate x quickly
# and robustly (without numerical problems), with very little memory usage.
# lsq_linop = scipy.sparse.linalg.LinearOperator((betavec0.size, betavec0.size),
#     matvec=l_jvp, rmatvec=l_vjp)

# we use a function to return the linear operator
def get_lsq_linop(bvsize, l_jvp, l_vjp):
    lsq_linop = scipy.sparse.linalg.LinearOperator((bvsize, bvsize),
        matvec=l_jvp, rmatvec=l_vjp)

def get_lsq_linop2(bvec, wh, xmat, geotargets, dw):
    l_diffs = lambda bvec: jax_targets_diff(bvec, wh, xmat, geotargets, dw)
    l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]
    l_vjp = lambda diffs: vjp(l_diffs, bvec)[1](diffs)
    lsq_linop = scipy.sparse.linalg.LinearOperator((bvec.size, bvec.size),
        matvec=l_jvp, rmatvec=l_vjp)



# %% OLD functions related to the jacobian and to Newton steps

# define lambda functions that each will take only one argument, and that
# will otherwise use items in the environment

# l_diffs: lambda function that gets differences from targets, as a vector
#   parameter: a beta vector
#     (In addition, it assumes existence of, and uses, values in the
#      environment: wh, xmat, geotargets, and dw. These do not change
#      within the loop.)
#   returns: vector of differences from targets
# l_diffs = lambda bvec: jax_targets_diff(bvec, wh, xmat, geotargets, dw)

# l_jvp: lambda function that evaluates the following jacobian vector product
#    (the dot product of a jacobian matrix and a vector)
#     matrix:
#      jacobian of l_diffs evaluated at the current bvec values,
#     vector:
#       differences from targets when l_diffs is evaluated at bvec
#   returns: a jacobian-vector-product vector
# This is used, in conjunction with l_vjp, to compute the step vector in the
# Newton method. It allows us to avoid computing the full jacobian, thereby
# saving considerable memory use and computation.
# l_jvp = lambda diffs: jvp(l_diffs, (bvec,), (diffs,))[1]

# l_vjp: lambda function that evaluates the following vector jacobian product
#    (the dot product of the transpose of a jacobian matrix and a vector)
#     matrix:
#      transpose of jacobian of l_diffs evaluated at the current bvec values,
#     vector:
#       differences from targets when l_diffs is evaluated at bvec
#   returns: a vector-jacobian-product vector
# Used with l_jvp - see above.
# l_vjp = lambda diffs: vjp(l_diffs, bvec)[1](diffs)

