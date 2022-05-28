module poissonFunctions

function geo_weights(beta, wh, xmat)
    # beta: coefficients, s x k
    # wh: weight for each household, h values
    # xmat: hh characteristics, h x k
    # betax = beta.dot(xmat.T)

    # bdims = size(geotargets)
    # beta = reshape(beta, size(geotargets))

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

function geo_targets(whs, xmat)
    whs' * xmat
end

function targ_pdiffs(calctargets, geotargets)
    diffs = calctargets - geotargets
    pdiffs = diffs ./ geotargets
    pdiffs
end

function sspd(calctargets, geotargets)
    # worry about what to do when a geotarget is zero
    diffs = calctargets - geotargets
    pdiffs = diffs ./ geotargets
    sqpdiffs = pdiffs.^2
    sspd = sum(sqpdiffs)
    sspd
end

function objfn(beta, wh, xmat, geotargets)
    beta = reshape(beta, size(geotargets))
    whs = geo_weights(beta, wh, xmat)
    calctargets = geo_targets(whs, xmat)
    obj = sspd(calctargets, geotargets)
    obj
end

end