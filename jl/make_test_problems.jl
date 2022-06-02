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


function get_rproblem()
    h = 10 # households
    k = 2  # characteristicss
    s = 3  # states

    wh = [43.45278 51.24605 39.08130 47.52817 44.98483 43.90340 37.35561 35.01735 45.55096 47.91773]' # transpose to column vec

    xmat = [0.113703411 0.609274733 0.860915384 0.009495756 0.666083758 0.693591292 0.282733584 0.292315840 0.286223285 0.186722790;
            0.6222994 0.6233794 0.6403106 0.2325505 0.5142511 0.5449748 0.9234335 0.8372956 0.2668208 0.2322259]' # transpose
    # xmat

    geotargets = [55.50609 73.20929;
                  61.16143 80.59494;
                  56.79071 75.41574]

    # calculate current national target values
    # wh * xmat

    # compare to geosum targets
    # sum(geotargets, dims=1)

    beta_opt = [
        -0.02736588 -0.03547895
        0.01679640  0.08806331
        -0.05385230  0.03097379]

    # whs optimal (to solve for)
    whs_opt =
    [13.90740 15.09438 14.45099
    16.34579 18.13586 16.76441
    12.42963 13.97414 12.67753
    15.60913 16.07082 15.84823
    14.44566 15.85272 14.68645
    14.06745 15.51522 14.32073
    11.70919 13.28909 12.35734
    11.03794 12.39991 11.57950
    14.90122 15.59650 15.05323
    15.72018 16.31167 15.88589]

    return (h=h, s=s, k=k, xmat=xmat, wh=wh, geotargets=geotargets, beta_opt=beta_opt, whs_opt=whs_opt)
  end


end


# %% info from old R problem

# targets_opt = whs_opt' * xmat
# targets_opt - geotargets


# %% r problem for comparison
# class rProblem:
#     """
#     Problem I solved in R, along with the optimal results obtained there.
#     """

#     def __init__(self):
#       self.wh = np.array([43.45278, 51.24605, 39.08130, 47.52817, 44.98483,
#                   43.90340, 37.35561, 35.01735, 45.55096, 47.91773])

#       # create some initial weights
#       # seed(1)
#       # r = np.random.normal(0, xsd, (h, k))

#       x1 = [0.113703411, 0.609274733, 0.860915384, 0.009495756, 0.666083758,
#             0.693591292, 0.282733584, 0.292315840, 0.286223285, 0.186722790]
#       x2 = [0.6222994, 0.6233794, 0.6403106, 0.2325505, 0.5142511, 0.5449748,
#             0.9234335, 0.8372956, 0.2668208, 0.2322259]
#       self.xmat = np.array([x1, x2]).T
#       self.h = self.xmat.shape[0]
#       self.k = self.xmat.shape[1]
#       self.s = 3
#       # geotargets is an s x k matrix of state-specific targets
#       self.geotargets = np.array(
#                   [[55.50609, 73.20929],
#                    [61.16143, 80.59494],
#                    [56.79071, 75.41574]])

# %% results from r problem - for checking against

# dw from get_dweights should be:
# 1.801604 1.635017 1.760851 1.365947 1.240773 1.325983

# delta when the beta matrix is 0 should be:
# 2.673062, 2.838026, 2.567032, 2.762710, 2.707713,
#     2.683379, 2.521871, 2.457231, 2.720219, 2.770873

# state weights when beta is 0 and we use the associated delta:
# > whs0
#           [,1]     [,2]     [,3]
#  [1,] 14.48426 14.48426 14.48426
#  [2,] 17.08202 17.08202 17.08202
#  [3,] 13.02710 13.02710 13.02710
#  [4,] 15.84272 15.84272 15.84272
#  [5,] 14.99494 14.99494 14.99494
#  [6,] 14.63447 14.63447 14.63447
#  [7,] 12.45187 12.45187 12.45187
#  [8,] 11.67245 11.67245 11.67245
#  [9,] 15.18365 15.18365 15.18365
# [10,] 15.97258 15.97258 15.97258

# targets when beta is 0
#          [,1]     [,2]
# [1,] 57.81941 76.40666
# [2,] 57.81941 76.40666
# [3,] 57.81941 76.40666

# sse_weighted 5.441764e-21

# $beta_opt_mat
#             [,1]        [,2]
# [1,] -0.02736588 -0.03547895
# [2,]  0.01679640  0.08806331
# [3,] -0.05385230  0.03097379

# $targets_calc
#          [,1]     [,2]
# [1,] 55.50609 73.20929
# [2,] 61.16143 80.59494
# [3,] 56.79071 75.41574

# $whs (optimal)
#           [,1]     [,2]     [,3]
#  [1,] 13.90740 15.09438 14.45099
#  [2,] 16.34579 18.13586 16.76441
#  [3,] 12.42963 13.97414 12.67753
#  [4,] 15.60913 16.07082 15.84823
#  [5,] 14.44566 15.85272 14.68645
#  [6,] 14.06745 15.51522 14.32073
#  [7,] 11.70919 13.28909 12.35734
#  [8,] 11.03794 12.39991 11.57950
#  [9,] 14.90122 15.59650 15.05323
# [10,] 15.72018 16.31167 15.88589

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
