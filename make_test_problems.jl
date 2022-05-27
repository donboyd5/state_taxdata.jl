using Statistics

h = 10 # households
k = 2  # characteristicss
s = 3  # states

wh = [43.45278 51.24605 39.08130 47.52817 44.98483 43.90340 37.35561 35.01735 45.55096 47.91773]

xmat = [0.113703411 0.609274733 0.860915384 0.009495756 0.666083758 0.693591292 0.282733584 0.292315840 0.286223285 0.186722790;
        0.6222994 0.6233794 0.6403106 0.2325505 0.5142511 0.5449748 0.9234335 0.8372956 0.2668208 0.2322259]'
xmat

geotargets = [55.50609 73.20929;
              61.16143 80.59494;
              56.79071 75.41574]

# calculate current national target values
wh * xmat

# compare to geosum targets
sum(geotargets, dims=1)

# whs optimal (to solve for)
whsopt =
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

targets_opt = whsopt' * xmat
targets_opt - geotargets



# %% OLD: vcat for xmat
x1 = [0.113703411 0.609274733 0.860915384 0.009495756 0.666083758 0.693591292 0.282733584 0.292315840 0.286223285 0.186722790]
x2 = [0.6222994 0.6233794 0.6403106 0.2325505 0.5142511 0.5449748 0.9234335 0.8372956 0.2668208 0.2322259]

# alt approach
# 10 households, 2 characteristics
xmat = vcat(x1, x2)'


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