module makeTestProblems

using Distributions, Random

function mtp(h, s, k)
    Random.seed!(123)
    xsd=.02
    ssd=.02
    pctzero=0.0
    # h = 8
    # k = 2
    # s = 3

    # create xmat
    d = Normal(0., xsd)
    r = rand(d, (h, k)) # xmat dimensions
    xmat = 100 .+ 20 .* r

    # create whs
    d = Normal(0., ssd)
    r = rand(d, (h, s)) # whs dimensions
    r[r .< -.9] .= -.9 # not sure about this
    whs = 10 .+ 10 .* (1 .+ r)
    ws = sum(whs, dims=1)
    wh = sum(whs, dims=2)
    geotargets = whs' * xmat
    targets = sum(geotargets, dims=1) # one target per k (characteristic)

    return (h=h, s=s, k=k, xmat=xmat, wh=wh, whs=whs, targets=targets, geotargets=geotargets)

end

end

# %% end of function mtp

# class Problem:
# """Problem elements."""

# def __init__(self, h, s, k, xsd=.02, ssd=.02, pctzero=0.0):

#   self.h = h
#   self.s = s
#   self.k = k

#   # prepare xmat
#   seed(1)
#   r = np.random.normal(0, xsd, (h, k))
#   # r = np.random.randn(h, k) / 100  # random normal)
#   xmean = 100 + 20 * np.arange(0, k)
#   xmat_full = xmean * (1 + r)
#   # inefficient, but...
#   xmat = xmat_full.copy()

#   if pctzero > 0:
#         # randomly set some elements of xmat to zero
#         np.random.seed(1)
#         indices = np.random.choice(np.arange(xmat.size), replace=False, size=int(xmat.size * pctzero))
#         xmat[np.unravel_index(indices, xmat.shape)] = 0
#         # if any rows have all zeros, put at least one nonzero element in
#         zero_rows = np.where(~xmat.any(axis=1))[0]
#         if zero_rows.size > 0:
#               xmat[zero_rows, :] = xmat_full[zero_rows, :]

#   self.xmat = xmat

#   r = np.random.normal(0, ssd, (h, s))
#   r[r < -.9] = -.9  # so that whs cannot be close to zero
#   self.whs = 10 + 10 * (1 + r)
#   self.wh = self.whs.sum(axis=1)
#   self.ws = self.whs.sum(axis=0)
#   self.geotargets = np.dot(self.whs.T, self.xmat)
#   self.targets = self.geotargets.sum(axis=0)

# def help():
#   print("The Problem class creates random problems of arbitrary size",
#         "and sparsity, for purposes of testing geosolve.\n")
#   print("It requires 3 integer arguments:",
#         "\th:\t\tnumber of households (tax returns, etc.)",
#         "\ts:\t\tnumber of states or other geographic areas",
#         "\tk:\t\tnumber of characteristics each household has, where",
#         "\t\t\t\tcharacteristics might be wages, dividends, etc.",
#         sep='\n')

#   print("A 4th argument to generate a sparse matrix, is pctzero, a float.")

#   print("\nIt creates an object with the following attributes:",
#         "\twh:\t\t\th-length vector of national weights for households",
#         "\txmat:\t\th x k matrix of characteristices (data) for households",
#         "\ttargets:\ts x k matrix of targets", sep='\n')

#   print("\nThe goal of geosolve is to find state weights that will",
#         "hit the targets while ensuring that each household's state",
#         "weights sum to its national weight.\n", sep='\n')
