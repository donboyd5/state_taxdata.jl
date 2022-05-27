using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions

# %% start by creating julia counterparts to selected python functions
# https://docs.julialang.org/en/v1/manual/functions/

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
whs = wh .* stateshares
sum(whs, dims=2)

# good, this takes a state-shares matrix and applies it to national weights to
# get each household's state weights

function state_weights(wh, stshares)
    # wh: weight for each household, h values
    # stshares: h x s
    whs = wh .* stateshares
    whs
end

state_weights(wh, stateshares)


function targets_diff(beta, wh, xmat, geotargets)

end



# %% selected old python functions

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


def jax_sspd(beta_object, wh, xmat, geotargets, diff_weights):
    diffs = jax_targets_diff(beta_object, wh, xmat, geotargets, diff_weights)
    sspd = jnp.square(diffs).sum()
    return sspd
