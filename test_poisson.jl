# %% Notes
# comment blocks: ctrl-k ctrl-c
# uncomment: clrl-k ctrl-u

# https://discourse.julialang.org/t/what-is-the-difference-between-zygote-vs-forwarddiff-and-reversediff/55789/3
# https://juliadiff.org/

# fix data viewer problem
# https://stackoverflow.com/questions/67698176/error-loading-webview-error-could-not-register-service-workers-typeerror-fai


# %% TODO:
# -- scaling function, module



# %% libraries
using LinearAlgebra
using Statistics
using Optim, LineSearches
using Optim: converged, maximum, maximizer, minimizer, iterations #some extra functions
using Zygote
using ForwardDiff
using Tables
using MINPACK
using LsqFit
using LeastSquaresOptim
using GalacticOptim
using GalacticOptimJL

# go into the package, into pkg, and enter build
# "C:\Users\donbo\.julia\dev\microweight"
# import Pkg; Pkg.dev(raw"C:\Users\donbo\.julia\dev\microweight")
# using microweight

# %% load local modules
include("jl/poisson_functions.jl")
import .poissonFunctions as psf

include("jl/make_test_problems.jl")
import .makeTestProblems as mtp

include("jl/getdata_functions.jl")
import .getTaxProblem as gtp

# include("jl/rproblem.jl")

# %% functions

function f(beta)
    psf.objfn(beta, wh, xmat, geotargets) # / obj_scale
end

function f!(out, beta)
  out = psf.objfn(beta, wh, xmat, geotargets) # / obj_scale
end

function g!(G, x)
  G .=  f'(x)
end

function f2(beta, pdummy)
  # pdummy added for GalacticOptim
  psf.objfn(beta, wh, xmat, geotargets)
end

function fvec(beta)
  # beta = reshape(beta, size(geotargets))
  psf.objvec(beta, wh, xmat, geotargets)
end

function fvec!(out, beta)
  # for LeastSquaresOptim inplace
  out .= psf.objvec(beta, wh, xmat, geotargets)
end

function gvec!(out, beta)
  out .= ForwardDiff.jacobian(x -> fvec(x), beta)
end


function jfvec(beta)
  ForwardDiff.jacobian(x -> fvec(x), beta)
end




# %% get a problem

tp = mtp.get_rproblem()

tp = mtp.mtp(10, 3, 2)
tp = mtp.mtp(100, 8, 4)
tp = mtp.mtp(1000, 12, 6)
tp = mtp.mtp(5000, 20, 8)
tp = mtp.mtp(10000, 25, 12)
tp = mtp.mtp(50000, 50, 20)
tp = mtp.mtp(100000, 75, 40)

tp = gtp.get_taxprob(9)


# %% unpack the problem
h = size(tp.xmat)[1]
# s = size(tp.geotargets)[1]
# k = size(tp.geotargets)[2]

# get data
xmat = tp.xmat
wh = tp.wh
geotargets = tp.geotargets

ibeta = zeros(length(geotargets))


# %% scaling
# scale targets and xmat
# scale_factors = xmat.sum(axis=0) / scale_goal  # get sum of each col of xmat
# xmat = jnp.divide(xmat, scale_factors)
# geotargets = jnp.divide(geotargets, scale_factors)
# scale_goal typically was 10 or 1000

# when all done, adjust results

# ? http://www.fitzgibbon.ie/optimization-parameter-scaling

obj_scale = 1.0

# wh scaling - multiplicative
wh_scale_goal = 100. # goal for the maxmimum
# whscale = wh_scale_goal / maximum(wh)  # median
whscale = wh_scale_goal / median(wh)
whscale = 1.0  # this overrides wh scaling
wh = wh .* whscale
quantile(vec(wh))

# make the initial adjustment to geotargets based on whscale
geotargets = geotargets .* whscale
quantile(vec(geotargets))
minimum(geotargets, dims=1)
maximum(geotargets, dims=1)

# xmat scaling multiplicative based on max abs of each col of geotargets
targ_goal = 10.0
# targscale = targ_goal ./ maximum(abs.(tp.geotargets), dims=1)
# targscale = targ_goal ./ sum(tp.geotargets, dims=1)
targscale = targ_goal ./ sum(xmat, dims=1)
geotargets = targscale .* geotargets
quantile(vec(geotargets))
xmat = targscale .* tp.xmat

# scale = (k / 1000.) ./ sum(abs.(tp.xmat), dims=1)
# targscale = 100000.0 ./ sum(xmat, dims=1)
# targscale = 0.1 ./ mean(abs.(xmat), dims=1) # mean avoids risk of 0 denominator
# adjust targets to be closer to a constant value
#targscale = 1. ./ maximum(geotargets, dims=1)
# targscale = 1.
# geotargets = targscale .* geotargets

# xmat = targscale .* tp.xmat

quantile(vec(xmat))


sum(xmat, dims=1)

# # %% examine if desired
# nattargs = sum(geotargets, dims=1) # national value for each target
# sum(wh)

# calcnat = sum(wh .* xmat, dims=1) # national value for each target at national weights
# calcdiffs = calcnat .- nattargs
# calcpdiffs = calcdiffs ./ nattargs
# quantile(vec(calcpdiffs))


# %% levenberg marquardt with LsqFit.lmfit
# https://github.com/sglyon/MINPACK.jl

# https://julianlsolvers.github.io/LsqFit.jl/latest/
# https://github.com/JuliaNLSolvers/LsqFit.jl/
# see Geodesic acceleration -- can I use curve_fit and this approach?
# re Geodesic acceleration see https://www.gnu.org/software/gsl/doc/html/nls.html

# also examine:
# https://juliapackages.com/p/leastsquaresoptim #  written with large scale problems in mind
# https://github.com/matthieugomez/LeastSquaresOptim.jl

# and possibly https://github.com/JuliaNLSolvers/NLsolve.jl

# Keyword arguments
# * `x_tol::Real=1e-8`: search tolerance in x
# * `g_tol::Real=1e-12`: search tolerance in gradient
# * `maxIter::Integer=1000`: maximum number of iterations
# * `min_step_quality=1e-3`: for steps below this quality, the trust region is shrinked
# * `good_step_quality=0.75`: for steps above this quality, the trust region is expanded
# * `lambda::Real=10`: (inverse of) initial trust region radius
# * `tau=Inf`: set initial trust region radius using the heuristic : tau*maximum(jacobian(df)'*jacobian(df))
# * `lambda_increase=10.0`: `lambda` is multiplied by this factor after step below min quality
# * `lambda_decrease=0.1`: `lambda` is multiplied by this factor after good quality steps
# * `show_trace::Bool=false`: print a status summary on each iteration if true
# * `lower,upper=[]`: bound solution to these limits

# returns
# param::P
# resid::R
# jacobian::J
# converged::Bool
# wt::W


ibeta = zeros(length(geotargets))
# ibeta = randn(length(tp.geotargets))

# LsqFit.lmfit(cost_function, [0.5, 0.5], Float64[])
fvec(ibeta)
f(ibeta)
# @time lsres = LsqFit.lmfit(fvec, ibeta, Float64[]; show_trace=true, maxIter=3)
# # 194.759903 secs

@time lsres = LsqFit.lmfit(fvec, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=50)
# Float64[] is a placeholder for weights for the parameters (differences)
# @time lsres = LsqFit.lmfit(fvec, lsres.param, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=100)
# 68.466341 secs for 3 iterations, 313.606478 secs for 30 iterations
sum(lsres.resid.^2)
f(lsres.param)
# fsolve stub10 5.782005 absmax, 242.1 f, 5342.655010 seconds
# lmfit stub10  5.78190591815, absmax, 89.89093155586703 f, 1136.002431 seconds

quantile(abs.(fvec(lsres.param)))

## end block

# 16 iter, 4.2 secs, 1.0141064462751911e-23

# * `x_tol::Real=1e-8`: search tolerance in x
# * `g_tol::Real=1e-12`: search tolerance in gradient
# * `maxIter::Integer=1000`: maximum number of iterations
# * `min_step_quality=1e-3`: for steps below this quality, the trust region is shrinked
# * `good_step_quality=0.75`: for steps above this quality, the trust region is expanded  (0, 1.0]
# * `lambda::Real=10`: (inverse of) initial trust region radius
# * `tau=Inf`: set initial trust region radius using the heuristic : tau*maximum(jacobian(df)'*jacobian(df))
# * `lambda_increase=10.0`: `lambda` is multiplied by this factor after step below min quality
# * `lambda_decrease=0.1`: `lambda` is multiplied by this factor after good quality steps
# * `show_trace::Bool=false`: print a status summary on each iteration if true
# * `lower,upper=[]`: bound solution to these limits
@time lsres = LsqFit.lmfit(fvec, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=5, min_step_quality=.5, good_step_quality=1.0)

# stub2 1.734587780229056e8
f(ibeta)
# baseline 10 iter: f 1.543305e+05 time 92:   1, 1.570074e+07, 2 2.211318e+06, 3 8.746149e05, 4 4.317217e+05, 5 4.317217e+05
# gtol 1e-4
# tau 10.0: BAD
# tau 1e9: BAD
# msq 1e-1: ok
# msq 0.5: ok
# msq 0.7  f  2.368598e+05
# gsq: 0.5 not as good as bl; 0.95 not as good
# lambda 5.0 ok
# lambda 15.0 f 4.482722e+05, 77 secs
# li 20.0 ok

## section divider
# prob9 time, f
# res0 57.308472, 7.819640e+04
# res1 61.556728, 7.819640e+04
# res2  58.648203, 7.819640e+04
@time lsres0 = LsqFit.lmfit(fvec, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=5)
@time lsres1 = LsqFit.lmfit(fvec, jfvec, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=5)
@time lsres2 = LsqFit.lmfit(fvec, jfvec, ibeta, Float64[]; show_trace=true, maxIter=25)

@time lsres0 = LsqFit.lmfit(fvec, results.zero, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=5)

# @time lsres3 = LsqFit.lmfit(fvec, lsres3.param, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=10)

# prob9 21 iter 225.801845 seconds
jfvec(ibeta)
f(lsres.param)

# Mads looong time 47266.37818506016

# fieldnames(lsres)
lsres.param
# LsqFit.lmfit(cost_function, [0.5, 0.5], Float64[], autodiff=:forward)
# LsqFit.LsqFitResult{Array{Float64,1},Array{Float64,1},Array{Float64,2},Array{Float64,1}}([2.0, 0.5], [4.44089e-16, 8.88178e-16, 1.77636e-15, 1.77636e-15, 1.77636e-15, 3.55271e-15, 3.55271e-15, 3.55271e-15, 3.55271e-15, 3.55271e-15], [-1.0 -0.0; -2.0 -0.0; … ; -9.0 -0.0; -10.0 -0.0], true, Float64[])


# %% levenberg marquardt with MADS
import Mads

function callback(x_best::AbstractVector, of::Number, lambda::Number)
	global callbacksucceeded
	callbacksucceeded = true
	# println("The callback function was called: $x_best, $of, $lambda")
  println(of, " ", lambda)
end

#, np_lambda=4, tolOF=1e-24, tolX=1e-24, tolG=1e-24

# @time madsm = Mads.minimize(f, ibeta)
@time madslm = Mads.naive_levenberg_marquardt(fvec, jfvec, ibeta; maxIter=200)
madslm.minimum

@time madslm = Mads.levenberg_marquardt(fvec, jfvec, ibeta;
  lambda = 10000.,
  show_trace=false, callbackiteration=callback)
madslm.minimum

@time madslm = Mads.levenberg_marquardt(fvec, jfvec, ibeta;
  #alwaysDoJacobian = true,
  # maxEval=2001, maxIter=2000, maxJacobians=2000,
  maxJacobians=20000,
  # tolOF=1e-99, tolX=1e-99, tolG=1e-99,
  show_trace=false, callbackiteration=callback)
madslm.minimum
f(ibeta)

@time madslm = Mads.levenberg_marquardt(fvec, jfvec, ibeta;
  #alwaysDoJacobian = true,
  # maxEval=2001, maxIter=2000, maxJacobians=2000,
  # maxJacobians=20000,
  # tolOF=1e-99, tolX=1e-99, tolG=1e-99,
  show_trace=false, callbackiteration=callback)

sum(fvec(madslm.minimizer).^2)


madslm.minimizer = ibeta

for i in 1:60
  println("iter group: ", i)
  @time madslm = Mads.levenberg_marquardt(fvec, jfvec, madslm.minimizer;
    #alwaysDoJacobian = true,
      # maxEval=2001, maxIter=2000, maxJacobians=2000,
        # maxJacobians=20000,
          # tolOF=1e-99, tolX=1e-99, tolG=1e-99,
            show_trace=false, callbackiteration=callback)
  println(madslm.minimum)
end

madslm.minimum
madslm.minimizer
f(madslm.minimizer)
madslm.Iter
# maxEval::Integer=1001, maxIter::Integer=100, maxJacobians::Integer=100
# tolX::Number=1e-4, tolG::Number=1e-6, tolOF::Number=1e-3, maxEval::Integer=1001, maxIter::Integer=100, maxJacobians::Integer=100,
# show_trace::Bool=false, alwaysDoJacobian::Bool=false,


# %% minpack least squares

# Available methods for the version where only f! is pased are:
#   :hybr: Modified version of Powell's algorithm. Uses MINPACK routine hybrd1
#   :lm: Levenberg-Marquardt. Uses MINPACK routine lmdif1
#   :lmdif: Advanced Levenberg-Marquardt (more options available with ;kwargs...). See MINPACK routine lmdif for more information
#   :hybrd: Advacned modified version of Powell's algorithm (more options available with ;kwargs...). See MINPACK routine hybrd for more information

#   Available methods for the version where both f! and g! are passed are:
#   :hybr: Advacned modified version of Powell's algorithm with user supplied Jacobian. Additional arguments are available via ;kwargs.... See MINPACK routine hybrj for more information
#   :lm: Advanced Levenberg-Marquardt with user supplied Jacobian. Additional arguments are available via ;kwargs.... See MINPACK routine lmder for more information

# - `algo::String`: the name of the algorithm used
# - `initial_x::Vector{Float64}`: the starting point
# - `x::Vector{Float64}`: The final value of `x`. When converged evaluating the objective at `x` should give zeros
# - `f::Vector{Float64}`: The function value at `x`
# - `return_code::Int`: The return code from MINPACK
# - `converged::Bool`: Whether or not the algorithm converged
# - `msg::String`: The message from MINPACK describing outcome
# - `trace::AlgoTrace`: If tracing was enabled, a detailed trace of all iterations

# /*       ftol is a nonnegative input variable. termination */
# /*         occurs when both the actual and predicted relative */
# /*         reductions in the sum of squares are at most ftol. */
# /*         therefore, ftol measures the relative error desired */
# /*         in the sum of squares. */

# /*       xtol is a nonnegative input variable. termination */
# /*         occurs when the relative error between two consecutive */
# /*         iterates is at most xtol. therefore, xtol measures the */
# /*         relative error desired in the approximate solution. */

# /*       gtol is a nonnegative input variable. termination */
# /*         occurs when the cosine of the angle between fvec and */
# /*         any column of the jacobian is at most gtol in absolute */
# /*         value. therefore, gtol measures the orthogonality */
# /*         desired between the function vector and the columns */
# /*         of the jacobian. */


# The trace prints:
#    f(x) inf-norm -- the infinity norm, or largest absolute value of the vector function, i.e., largest percentage difference from target
#    Step 2-norm -- the Euclidean norm, or p-norm where p=2: the square root of sum of squared errors, APPARENTLY after any scaling MINPACK does (?)
#      This ?? APPEARS to be the CHANGE in the 2-norm from prior iteration, and not its level value (I think)
#    Step time -- time in seconds for the step

# kwargs: iterations, xtol, gtol

# mp1 = fsolve(fvec!, ibeta, show_trace=true, method=:lm)

@time mp2 = fsolve(fvec!, gvec!, ibeta, show_trace=true, method=:lm, ftol=1e-6, xtol=1e-6, iterations=200)

# stub10 5.782005 absmax, 242.1 f, 5342.655010 seconds fsolve
f(mp2.x)
# @time mp2hybr = MINPACK.fsolve(fvec!, gvec!, ibeta, show_trace=true, method=:hybr, iterations=200)
stop

# useful info:?
@__FILE__
@__DIR__
@__MODULE__
pwd()

# @time mp2 = fsolve(fvec!, gvec!, ibeta, show_trace=true, method=:lm, ftol=1e-12, xtol=1e-12, gtol=1e-12, iterations=200)
# 77 iter
# 33     2.842458e+00     4.642699e+07        14.892000
# 34     1.352172e-01     3.507077e+07        14.415000
# 35     3.087479e-04     1.817047e+08        14.271000
# 36     8.797618e-09     5.161292e+08        14.315000
# 37     8.851874e-12     2.599300e+09        14.320000
# 38     2.503986e-11     9.812514e+09        14.426000
# ...
# 76     2.212671e-12     1.279096e-13         0.117000
# 77     1.993709e-12     2.038150e-14         0.107000
# 608.801005 seconds (141.90 k allocations: 1.394 TiB, 6.70% gc time)

# stub 2
# @time mp2 = fsolve(fvec!, gvec!, ibeta, show_trace=true, method=:lm, ftol=1e-36, xtol=1e-36, gtol=1e-36, iterations=200)
# Iter     f(x) inf-norm    Step 2-norm      Step time
# ------   --------------   --------------   --------------
#      1     2.107000e+03     0.000000e+00         1.633000
#      2     2.010310e+03     1.765999e-01        45.426000
#      3     1.890495e+03     6.097582e-01        39.905000
#      4     1.595539e+03     1.956376e+00        38.810000
#      5     8.529553e+02     5.523061e+00        38.324000
#      6     5.149884e+02     1.121274e+01        39.422000
#      7     1.208223e+04     3.280884e+01        39.242000
#      8     2.291551e+02     3.316581e+01         0.411000
#      9     9.264611e+01     1.202652e+00        38.460000
#     10     6.790096e+02     4.133745e+00        38.402000
#     11     1.174201e+02     3.052218e+00         0.226000
#     12     7.964013e+01     1.361486e-01        38.495000
#     13     7.464847e+01     6.134337e-01        38.832000
#     14     4.821147e+02     2.574159e+00        38.443000
#     15     6.035718e+01     1.815423e+00         0.226000
#     16     5.887238e+01     3.243595e-01        38.387000
#     17     1.482471e+02     1.308891e+00        38.340000
#     18     5.169034e+01     9.502969e-01        38.656000
#     19     8.047411e+01     3.093712e+00        38.844000
#     20     1.950647e+02     2.723394e+00        38.772000
#     21     4.272415e+01     2.094580e+00         0.227000
#     22     3.887941e+01     6.304415e-02        38.781000
#     23     3.483526e+01     2.687032e-01        38.666000
#     24     7.384722e+01     1.223642e+00        41.447000
#     25     3.310906e+01     7.971934e-01         0.336000
#     26     2.949120e+01     2.067931e-01        44.513000
#     27     2.656586e+01     2.238721e-01        40.590000
#     28     2.599051e+01     2.254452e-01        39.671000
#     29     2.464196e+01     9.331398e-01        39.429000
#     30     4.898500e+01     9.575015e-01        43.288000
#     31     2.414642e+01     4.838003e-01         0.222000
#     32     2.349321e+01     7.246711e-02        40.822000
#     33     2.201458e+01     2.831465e-01        41.772000
#     34     3.357772e+01     2.520746e-01        39.407000
#     35     2.182064e+01     1.696352e-01         0.235000
#     36     2.138795e+01     3.028531e-02        39.898000
#     37     2.045927e+01     1.251490e-01        40.058000
#     38     1.948496e+01     1.252465e-01        38.457000
#     39     1.706845e+01     4.880425e-01        39.542000
#     40     1.842015e+01     1.236723e-01         0.217000
#     41     1.587496e+01     4.592618e-01        39.750000
#     42     1.302957e+01     4.107799e-01        40.066000
#     43     4.177000e+01     1.465628e+00        38.521000
#     44     1.238574e+01     1.151184e+00         0.219000
#     45     1.099384e+01     6.306516e-02        36.939000
#     46     9.111326e+00     2.472342e-01        39.572000
#     47     5.880642e+00     2.474693e-01        39.637000
#     48     1.895909e+01     9.838352e-01        39.751000
#     49     5.577687e+00     4.585844e-01         0.234000
#     50     7.819505e+00     3.915322e-01        38.855000
#     51     5.916240e+00     3.561281e-01        37.465000
#     52     4.587411e+00     3.190719e-01        40.044000
#     53     4.077534e+00     2.813337e-01        39.485000
#     54     3.560254e+00     2.537409e-01        38.624000
#     55     3.358038e+00     2.305660e-01        42.142000
#     56     4.792417e+00     2.033571e-01        39.788000
#     57     3.043580e+00     1.863337e-01        36.572000
#     58     2.923033e+00     6.872914e-01        39.783000
#     59     3.446651e+00     6.353962e-01        42.957000
#     60     2.887156e+00     2.684509e-01         0.218000
#     61     2.790770e+00     3.110117e-01        39.050000
#     62     2.856295e+00     1.314029e-01         0.220000
#     63     2.788690e+00     1.530152e-01        39.230000
#     64     2.712347e+00     1.514911e-01        37.092000
#     65     2.622707e+00     1.516296e-01        38.662000
#     66     2.515335e+00     1.454643e-01        39.204000
#     67     2.391459e+00     1.221810e-01        41.350000
#     68     2.330903e+00     2.559918e-02        39.706000
#     69     2.192401e+00     7.516562e-02        40.024000
#     70     2.308313e+00     4.839712e-02         0.220000
#     71     2.263243e+00     9.247735e-03        39.309000
#     72     2.181773e+00     2.447766e-02        39.381000
#     73     2.138525e+00     6.567108e-03        39.805000
#     74     2.056414e+00     1.879930e-02        38.539000
#     75     2.102745e+00     5.609542e-03         0.134000
#     76     2.036016e+00     1.195803e-02        39.546000
#     77     2.005070e+00     2.602513e-03        40.408000
#     78     1.944914e+00     8.549918e-03        39.543000
#     79     1.916757e+00     1.922456e-03        39.419000
#     80     1.861722e+00     6.588728e-03        39.130000
#     81     1.835688e+00     1.515636e-03        39.466000
#     82     1.784530e+00     5.348486e-03        39.148000
#     83     1.812773e+00     1.623108e-03         0.134000
#     84     1.768940e+00     3.892804e-03        38.958000
#     85     1.747756e+00     9.195409e-04        39.306000
#     86     1.705937e+00     3.368314e-03        37.164000
#     87     1.685562e+00     8.030819e-04        38.655000
#     88     1.645331e+00     2.976356e-03        38.626000
#     89     1.665820e+00     7.811562e-04         0.135000
#     90     1.627810e+00     2.636421e-03        38.701000
#     91     1.609089e+00     6.339067e-04        38.590000
#     92     1.572168e+00     2.381590e-03        38.394000
#     93     1.553956e+00     5.743307e-04        39.287000
#     94     1.518160e+00     2.169446e-03        38.736000
#     95     1.500522e+00     5.234506e-04        38.713000
#     96     1.465962e+00     1.984851e-03        40.120000
#     97     1.448995e+00     4.786397e-04        39.527000
#     98     1.415818e+00     1.819946e-03        40.300000
#     99     1.383881e+00     1.713553e-03        40.296000
#    100     1.352878e+00     1.618289e-03        38.786000
#    101     1.322852e+00     1.529671e-03        39.179000
#    102     1.290598e+00     1.785629e-03        38.833000
#    103     1.259567e+00     1.677981e-03        39.033000
#    104     1.201024e+00     5.975090e-03        39.063000
#    105     1.237731e+00     2.335182e-03         0.136000
#    106     1.195744e+00     3.102494e-03        39.150000
#    107     1.156254e+00     2.838703e-03        39.018000
#    108     1.118477e+00     2.609201e-03        38.941000
#    109     1.047260e+00     8.920123e-03        39.308000
#    110     1.089966e+00     3.088488e-03         0.134000
#    111     1.035158e+00     5.361477e-03        39.431000
#    112     9.848621e-01     4.731886e-03        39.197000
#    113     9.297076e-01     5.582487e-03        39.367000
#    114     8.785670e-01     4.884155e-03        38.719000
#    115     7.836880e-01     1.522556e-02        39.408000
#    116     8.471317e-01     6.296221e-03         0.141000
#    117     7.856801e-01     6.591315e-03        39.226000
#    118     7.221131e-01     7.343599e-03        39.540000
#    119     6.646685e-01     6.116553e-03        39.653000
#    120     6.024329e-01     6.938246e-03        39.185000
#    121     5.470218e-01     5.641566e-03        39.392000
#    122     5.455600e-01     6.540846e-03        39.545000
#    123     4.599914e-01     1.444607e-03        39.221000
#    124     5.765457e-01     5.589724e-03        39.301000
#    125     4.449354e-01     2.928606e-03         0.136000
#    126     4.156074e-01     1.482979e-03        39.244000
#    127     3.851095e-01     1.644610e-03        40.029000
#    128     3.570532e-01     1.404694e-03        40.175000
#    129     3.271090e-01     1.574818e-03        39.987000
#    130     2.998927e-01     1.315809e-03        39.643000
#    131     2.706176e-01     1.497219e-03        40.567000
#    132     2.444347e-01     1.214948e-03        39.291000
#    133     2.549114e-01     1.408396e-03        40.249000
#    134     2.032226e-01     3.098050e-04        39.968000
#    135     2.501996e-01     1.193442e-03        39.852000
#    136     1.941457e-01     5.035867e-04         0.134000
#    137     1.773045e-01     4.887152e-04        39.649000
#    138     1.627576e-01     5.549010e-04        39.277000
#    139     1.517488e-01     4.490074e-04        39.115000
#    140     1.394009e-01     5.180797e-04        40.689000
#    141     1.440649e-01     4.965178e-04        40.074000
#    142     1.512652e-01     4.730328e-04        40.688000
#    143     1.590360e-01     4.473271e-04        39.607000
#    144     1.674313e-01     4.190597e-04        38.989000
#    145     7.974132e-02     8.428051e-05        39.698000
#    146     1.502136e-01     3.123569e-04        39.799000
#    147     7.697165e-02     1.910137e-04         0.233000
#    148     7.183893e-02     4.906362e-05        39.439000
#    149     9.383175e-02     1.836231e-04        40.468000
#    150     6.840758e-02     8.061429e-05         0.220000
#    151     6.147013e-02     7.985612e-05        40.608000
#    152     5.440139e-02     7.575537e-05        38.842000
#    153     4.721475e-02     7.140916e-05        38.784000
#    154     4.451646e-02     6.681041e-05        39.470000
#    155     4.575059e-02     6.193692e-05        39.114000
#    156     4.701090e-02     5.676408e-05        38.885000
#    157     4.828725e-02     5.126121e-05        39.039000
#    158     4.956738e-02     4.538240e-05        39.125000
#    159     1.258078e-02     1.035521e-05        38.280000
#    160     2.388637e-02     2.958021e-03        38.663000
#    161     4.447447e-03     2.920395e-03         0.487000
#    162     7.711125e-03     2.389337e-03        38.425000
#    163     3.397619e-03     2.401284e-03         0.577000
#    164     8.706054e-04     2.329865e-04        38.406000
#    165     5.616337e-11     9.053091e-04        38.393000
#    166     1.734261e-12     7.195212e-03        38.079000
#    167     3.127598e-12     2.147383e-02        38.004000
#    168     3.839090e-12     9.791390e-03         0.133000
#    169     3.957672e-12     1.308766e-03         0.131000
#    170     3.424053e-12     7.703360e-05         0.131000
#    171     2.712562e-12     3.604262e-06         0.155000
#    172     2.238234e-12     2.134538e-07         0.135000
#    173     1.694354e-12     2.349607e-08         0.130000
#    174     3.068307e-12     1.528388e-08        38.599000
#    175     2.119652e-12     9.594680e-09         0.137000
#    176     1.752780e-12     3.180169e-10         0.136000
#    177     1.662341e-12     1.810282e-11        39.170000
#    178     1.694354e-12     4.988755e-12        38.883000
#    179     2.445752e-12     7.935315e-13        39.268000
#    180     3.809445e-12     5.449577e-13         0.130000
#    181     3.839090e-12     1.886014e-14         0.134000
#    182     1.635928e-12     1.886014e-16         0.150000
#    183     1.694354e-12     1.025231e-18         0.138000
#    184     1.674879e-12     1.027094e-19         0.145000
#    185     1.694354e-12     1.044986e-20        38.789000
#    186     1.713829e-12     1.859825e-21        38.834000
#    187     1.713829e-12     5.808200e-22         0.141000
#    188     1.713829e-12     1.562066e-22        41.133000
#    189     2.534689e-12     4.173598e-23         0.133000
#    190     1.681671e-12     2.613253e-23         0.135000
#    191     1.720330e-12     6.086699e-25         0.223000
#    192     1.674879e-12     2.327781e-26         0.618000
#    193     1.701001e-12     7.626428e-28         0.325000
#    194     1.674879e-12     3.540342e-28         0.317000
# 5843.449044 seconds (25.70 M allocations: 13.947 TiB, 6.40% gc time, 0.10% compilation time)


mp2.algo
mp2.initial_x
mp2.x
# f(mp1.x)
f(mp2.x)
mp2.f
# sum(mp1.f.^2)
sum(mp2.f.^2)
mp2.trace

@time mp3 = fsolve(fvec!, gvec!, mp2.x, show_trace=true, method=:lm, ftol=1e-36, xtol=1e-36, gtol=1e-36, iterations=20)
f(mp3.x)
quantile(mp3.f, (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))



# %% curve_fit

function geotargs(geotargets, beta)
  # assumes wh and xmat are in the environment
  beta = reshape(beta, size(tp.geotargets))
  whs = psf.geo_weights(beta, wh, xmat)
  vec(psf.geo_targets(whs, xmat))
end

# fit = curve_fit(model, xdata, ydata, p0)
fit = curve_fit(geotargs, vec(geotargets), vec(geotargets), ibeta)
fit.param
f(fit.param)
sum(fit.resid.^2)

ibeta = zeros(length(geotargets))
fit = curve_fit(geotargs, vec(geotargets), vec(geotargets), ibeta; autodiff=:forwarddiff, show_trace=true, maxIter=500)

using LinearAlgebra, Zygote

function Avv!(dir_deriv,beta,v)
    for i=1:length(geotargets)
        dir_deriv[i] = transpose(v) * Zygote.hessian(z->geotargs(vec(geotargets)[i], z), beta) * v
    end
end

fit2 = curve_fit(geotargs, vec(geotargets), vec(geotargets), ibeta; avv! = Avv!,lambda=0, min_step_quality = 0)

# function Avv!(dir_deriv,p,v)
#     for i=1:length(xdata)
#         dir_deriv[i] = transpose(v) * Zygote.hessian(z->model(xdata[i],z),p) * v
#     end
# end

# %% try to speed up LsqFit.lmfit
# https://github.com/JuliaNLSolvers/LsqFit.jl/

# %% inplace: It is possible to either use an in-place model, or an in-place model and an in-place Jacobian. It might be pertinent to use this feature when curve_fit is slow, or consumes a lot of memory.

# a two-parameter exponential model
# x: array of independent variables
# p: array of model parameters
# model(x, p) will accept the full data set as the first argument `x`.
# This means that we need to write our model function so it applies
# the model to the full dataset. We use `@.` to apply the calculations
# across all rows.
@. model(x, p) = p[1]*exp(-x*p[2]) # this is simply a function with 2 parameters p that predicts y based on x, which will have length(x)
# in my case, it predicts geotargets based on params beta
# or really, it predicts geo targets based on params beta, and data xmat and wh

# some example data
# xdata: independent variables
# ydata: dependent variable
xdata = range(0, stop=10, length=20)
typeof(xdata)

ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
p0 = [0.5, 0.5]

model_inplace(F, x, p) = (@. F = p[1] * exp(-x * p[2]))

function jacobian_inplace(J::Array{Float64,2},x,p)
        @. J[:,1] = exp(-x*p[2])
        @. @views J[:,2] = -x*p[1]*J[:,1]
    end
fit = curve_fit(model_inplace, jacobian_inplace, xdata, ydata, p0; inplace = true)
fit.param

# %% Geodesic acceleration
# see Geodesic acceleration -- can I use curve_fit and this approach?
# https://arxiv.org/pdf/1010.1449.pdf
# curve_fit calls lmfit https://github.com/JuliaNLSolvers/LsqFit.jl/blob/master/src/curve_fit.jl
# lmfit eventually calls levenberg_marquardt, line 68

# To enable it [Geodesic acceleration], one needs to specify the function computing the directional second derivative of the function that is fitted,
# as the avv! parameter. It is also preferable to set lambda and min_step_qualityto 0:
# curve_fit(model, xdata, ydata, p0; avv! = Avv!,lambda=0, min_step_quality = 0)

# Avv! must have the following form:
  # p is the array of parameters
  # v is the direction in which the direction is taken [???]
  # dir_deriv is the output vector (the function is necessarily in-place)

function Avv!(dir_deriv,p,v)
  v1 = v[1]
  v2 = v[2]
  for i=1:length(xdata)
      #compute all the elements of the Hessian matrix
      h11 = 0
      h12 = (-xdata[i] * exp(-xdata[i] * p[2]))
      #h21 = h12
      h22 = (xdata[i]^2 * p[1] * exp(-xdata[i] * p[2]))

      # manually compute v'Hv. This whole process might seem cumbersome, but
      # allocating temporary matrices quickly becomes REALLY expensive and might even
      # render the use of geodesic acceleration terribly inefficient
      dir_deriv[i] = h11*v1^2 + 2*h12*v1*v2 + h22*v2^2

  end
end

m1 = curve_fit(model, xdata, ydata, p0)
m2 = curve_fit(model, xdata, ydata, p0; avv! = Avv!,lambda=0, min_step_quality = 0)
m1.param
m2.param

function Avv!(dir_deriv,p,v)
  for i=1:length(xdata)
      dir_deriv[i] = transpose(v) * Zygote.hessian(z -> model(xdata[i], z), p) * v  # z stand in for parameters
  end
end

m3 = curve_fit(model, xdata, ydata, p0; avv! = Avv!,lambda=0, min_step_quality = 0)
m3.param

m3 = curve_fit(w, xdata, ydata, p0; avv! = Avv!,lambda=0, min_step_quality = 0)
m3.param

model(xdata[1], z)

function f7(x)
  x.^2
end
f7(3.)
f7([1., 2., 3.])


function Avv!(dir_deriv, p, v)
  for i=1:length(p)
      dir_deriv[i] = transpose(v) * Zygote.hessian(z -> f7(z), p) * v
  end
end

function Avv!(dir_deriv, p, v)
  dir_deriv = transpose(v) * Zygote.hessian(z -> f7(z), p) * v
end

res1 = LsqFit.lmfit(f7, [.1, .7, .3], Float64[]; autodiff=:forwarddiff, show_trace=true)
res1.param

res2 = LsqFit.lmfit(f7, [.1, .7, .3], Float64[]; avv! = Avv!,lambda=0, min_step_quality = 0, show_trace=true)
res2.param


res3 = LsqFit.lmfit(f7, [.1, .7, .3], Float64[]; autodiff=:forwarddiff, avv! = Avv!,lambda=0, min_step_quality = 0, show_trace=true)
res3.param

res1.param
res2.param
res3.param




ibeta = zeros(length(geotargets))
@time lsres1 = LsqFit.lmfit(fvec, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=10)

function Avv!(dir_deriv, beta, v)
  for i=1:length(geotargets)
      dir_deriv[i] = transpose(v) * Zygote.hessian(z -> geotargs(geotargets[i], z), beta) * v
  end
end

beta = ibeta

function geotargs(geotargets, beta)
  # assumes wh and xmat are in the environment
  beta = reshape(beta, size(tp.geotargets))
  whs = psf.geo_weights(beta, wh, xmat)
  vec(psf.geo_targets(whs, xmat))
end

geotargs(11., ibeta)

curve_fit(geotargs, geotargets, ydata, ibeta; avv! = Avv!,lambda=0, min_step_quality = 0)

@time lsres2 = LsqFit.lmfit(fvec2, ibeta, Float64[]; avv! = Avv!,lambda=0, min_step_quality = 0, show_trace=true, maxIter=10)
# @time lsres3 = LsqFit.lmfit(fvec, lsres3.param, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=10)

# prob9 21 iter 225.801845 seconds

f(lsres1.param)



geotargs(geotargets, ibeta)

function fvec2(beta)
  # beta = reshape(beta, size(geotargets))
  psf.objvec(beta, wh, xmat, geotargets)
end

reshape(ibeta, size(geotargets))

Zygote.hessian(fvec, 10.)
fvec(10.)
fvec([10., 5.])

ibeta


# %% examine results
fvec([10.])
# dump(res)
res = res12
beta_o = reshape(minimizer(res), size(geotargets))

# levenberg marquardt results
res = madslm.minimizer
res = lsres.param
res = results.zero
res = lsores.minimizer
res = mp2.x
f(res)
# res = fit.param

beta_o = reshape(res, size(geotargets))


# calc results
whs_o = psf.geo_weights(beta_o, wh, xmat)
whdiffs = sum(whs_o, dims=2) .- wh
whpdiffs = whdiffs ./ wh
quantile(vec(whpdiffs), (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))

geotarg_o = psf.geo_targets(whs_o, xmat)
geotargets
# quantile(vec(abs.(tp.geotargets)))
gtpdiffs = psf.targ_pdiffs(geotarg_o, geotargets)
quantile(vec(gtpdiffs), (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))

psf.sspd(geotarg_o, geotargets)
rmse = (psf.sspd(geotarg_o, geotargets) / length(geotargets))^.5

# now return to unscaled values
whs_ou = whs_o ./ whscale
wh_ou = sum(whs_ou, dims=2)
quantile(vec((wh_ou .- tp.wh) ./ tp.wh), (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))

getarg_ou = geotargets ./ targscale ./ whscale
tp.geotargets
quantile(vec((getarg_ou .- tp.geotargets) ./ tp.geotargets), (0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0))


# %% SEPARATOR %%################################

# %% try LeastSquaresOptim
# using LeastSquaresOptim
# least squares optimizers Dogleg() and LevenbergMarquardt()
# solvers LeastSquaresOptim.QR() or LeastSquaresOptim.Cholesky() for dense jacobians
# For dense jacobians, the default option is Dogleg(QR()).
# For sparse jacobians, the default option is LevenbergMarquardt(LSMR())
# optimize(rosenbrock, x0, Dogleg(LeastSquaresOptim.QR()))

# You can also add the options : x_tol, f_tol, g_tol, iterations, Δ (initial radius),
# autodiff (:central to use finite difference method or :forward to use ForwardDiff package)
#  as well as lower / upper arguments to impose boundary constraints.

# prob9
# inplace forward, Dogleg, QR, 29 iter 273.723267 seconds
# inplace forward, Dogleg, Cholesky,  rank deficient
# inplace forward, Dogleg, LSMR, 30 NOT yet conveged (2106591.468742), 203 secs
# inplace forward, LevenbergMarquardt, QR, 30 not converged iter 298 seconds

@time lsores = LeastSquaresOptim.optimize(fvec, ibeta, Dogleg(LeastSquaresOptim.QR()),
 autodiff = :forward, show_trace=true, iterations=30)

# djb this seems good, maybe with 200 iterations??
@time lsores = LeastSquaresOptim.optimize(fvec, ibeta, LevenbergMarquardt(LeastSquaresOptim.LSMR()),
  autodiff = :forward, show_trace=true, iterations=100) # 626.6 secs; next 50 859; next 50 864.9; next 50 1000.307757; next 50 1083.807849

@time lsores2 = LeastSquaresOptim.optimize(fvec, lsores2.minimizer, LevenbergMarquardt(LeastSquaresOptim.LSMR()),
  autodiff = :forward, show_trace=true, iterations=50)

  lsores2.minimizer
#  @time  lsores = LeastSquaresOptim.optimize(fvec, ibeta, Dogleg(LeastSquaresOptim.Cholesky()),
#  autodiff = :forward, show_trace=true, iterations=5)

# inplace supposed to be faster and uses less memory
ibeta = zeros(length(tp.geotargets))
@time lsores = LeastSquaresOptim.optimize!(
  LeastSquaresOptim.LeastSquaresProblem(x = ibeta, f! = fvec!, output_length = length(ibeta), autodiff = :forward),
  LeastSquaresOptim.Dogleg(LeastSquaresOptim.QR()),
  show_trace=true,
  iterations=500)
lsores.minimizer

f(lsores.minimizer)

# %% run methods that do not require a gradient
res1 = Optim.optimize(f, ibeta, NelderMead(),
Optim.Options(g_tol = 1e-12, iterations = 10, store_trace = true, show_trace = true))

res2 = Optim.optimize(f, ibeta, SimulatedAnnealing(),
  Optim.Options(g_tol = 1e-4, iterations = 100))


# %% run methods that require a gradient with forward auto differentiation
res3 = Optim.optimize(f, ibeta, BFGS(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
   autodiff = :forward) # gets slow with BIG problems

# two ways to do it
res4 = Optim.optimize(f, ibeta, LBFGS(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward)
# 1897.9 4.125432e-14

 res4a = optimize(beta -> psf.objfn(beta, wh, xmat, geotargets), ibeta, LBFGS(); autodiff = :forward)


# CG possibly best!
# very good after 2 iterations ~4 mins, almost perfect after 3, 4.8 mins
res5 = Optim.optimize(f, lsres.param, ConjugateGradient(),
Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
 autodiff = :forward)
# 1102.0 4.125609e-14 seemingly best

 # this seems really good after 3 iterations
 res6 = Optim.optimize(f, ibeta, GradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward) # seems to become very slow as problem size increases
# 1910.7 3.078448e-11

# really good after 5 iterations
res7 = Optim.optimize(f, ibeta, MomentumGradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward)
# 1602.8 3.078448e-11

# really good after 3 iterations 562 secs
res8 = Optim.optimize(f, ibeta, AcceleratedGradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward)
# 1924.9 3.078448e-11

# hessian required
# https://julianlsolvers.github.io/Optim.jl/stable
# https://julianlsolvers.github.io/Optim.jl/v0.9.3/algo/autodiff/
# https://github.com/JuliaNLSolvers/Optim.jl/tree/v0.4.5
# td = TwiceDifferentiable(f, ibeta; autodiff = :forward)
# res9 = optimize(td, vec(ibeta), Newton())
# Newton requires vector input; TOO slow with big problems
res9 = Optim.optimize(f, ibeta, Newton(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward) # seems to get slow as problem gets large

res10 = Optim.optimize(f, ibeta, NewtonTrustRegion(),
  Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true);
  autodiff = :forward)

ibeta = zeros(length(geotargets))
td = TwiceDifferentiable(f, ibeta; autodiff = :forward)
opts = Optim.Options(g_tol = 1e-6, iterations = 10, store_trace = true, show_trace = true)
res10a = Optim.optimize(td, ibeta, Newton(), opts)
res10b = Optim.optimize(td, ibeta, NewtonTrustRegion(), opts)
minimizer(res10a)

f(minimizer(res10a.minimizer))

# %% run reverse auto differentiation
f(ibeta)
res11 = Optim.optimize(f, g!, ibeta, LBFGS(),
  Optim.Options(iterations = 100, store_trace = true, show_trace = true))

res11a = Optim.optimize(f, g!, minimizer(res11), LBFGS(),
  Optim.Options(iterations = 2000, store_trace = true, show_trace = true))

res11a = Optim.optimize(f, g!, sol6a.minimizer, LBFGS(),
  Optim.Options(iterations = 2000, store_trace = true, show_trace = true))

# cg still seems best
using LineSearches
# alphaguess = LineSearches.InitialQuadratic() and linesearch = LineSearches.MoreThuente()  investigate
# HagerZhang MoreThuente BackTracking StrongWolfe Static
f(ibeta)
# this seems good 6/1/2022
res12 = Optim.optimize(f, g!, ibeta, ConjugateGradient(eta=0.01; alphaguess = LineSearches.InitialConstantChange(), linesearch = LineSearches.HagerZhang()),
  Optim.Options(g_tol = 1e-6, iterations = 1_000, store_trace = true, show_trace = true))
# 4.669833e+03 after 10k
# 2.030909e+03 after 20k
# 1.173882e+03 after 30k
#  after 40k
#  after 50k
res12a = Optim.optimize(f, g!, minimizer(res12a), ConjugateGradient(eta=0.01; alphaguess = LineSearches.InitialConstantChange(), linesearch = LineSearches.HagerZhang()),
  Optim.Options(g_tol = 1e-6, iterations = 10_000, store_trace = true, show_trace = true))
  # minimizer(res12)
  # minimizer(res12a)

# ConjugateGradient(;linesearch = LLineSearches.HagerZhang())
# ConjugateGradient(;linesearch = LineSearches.MoreThuente())
res12a = Optim.optimize(f, g!, minimizer(res11a), ConjugateGradient(eta=0.01),
  Optim.Options(g_tol = 1e-6, iterations = 10000, store_trace = true, show_trace = true))

res12b = Optim.optimize(f, g!, minimizer(res12a), ConjugateGradient(eta=0.01;linesearch = LineSearches.MoreThuente()),
  Optim.Options(g_tol = 1e-6, iterations = 10000, store_trace = true, show_trace = true))

using TimerOutputs
prob = Optim.UnconstrainedProblems.examples["Rosenbrock"]
const to = TimerOutput()

f1(x    ) =  @timeit to "f1"  prob.f(x)
g1!(x, g1) =  @timeit to "g1!" prob.g!(x, g)
h1!(x, h1) =  @timeit to "h1!" prob.h!(x, h)

begin
reset_timer!(to)
@timeit to "Trust Region" begin
    res = Optim.optimize(f1, g1!, h1!, prob.initial_x, NewtonTrustRegion())
end
show(to; allocations = false)
end


res12a = Optim.optimize(f, g!, sol6a.minimizer, ConjugateGradient(eta=0.01),
  Optim.Options(g_tol = 1e-6, iterations = 2000, store_trace = true, show_trace = true))

res13 = Optim.optimize(f, g!, ibeta, GradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

res13a = Optim.optimize(f, g!, minimizer(res12a), GradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 2000, store_trace = true, show_trace = true))

res14 = Optim.optimize(f, g!, ibeta, MomentumGradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 10000, store_trace = true, show_trace = true))

res15 = Optim.optimize(f, g!, ibeta, AcceleratedGradientDescent(),
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

# res16 = Optim.optimize(f, g!, ibeta, Newton(),
#   Optim.Options(g_tol = 1e-6, iterations = 5, store_trace = true, show_trace = true);
#   autodiff = :forward) # seems to get slow as problem gets large

res17 = Optim.optimize(f, g!, ibeta, ConjugateGradient(eta=0.01),
  Optim.Options(iterations = 1000, store_trace = true, show_trace = true))

res17a = Optim.optimize(f, g!, minimizer(res11a), ConjugateGradient(eta=0.01),
  Optim.Options(iterations = 2000, store_trace = true, show_trace = true))


# res18 = optimize(f, g!, ibeta, Optim.KrylovTrustRegion(),
#   Optim.Options(iterations = 10, store_trace = true, show_trace = true))

# J = JacVec(f, u::AbstractArray; autodiff=true)
# lmfit prob1 239.627257  f 4.169626e+02  73 iters


# %% accelerate CG or other --------------------------------
# ConjugateGradient(eta=0.01; alphaguess = LineSearches.InitialConstantChange(), linesearch = LineSearches.HagerZhang())

res12 = Optim.optimize(f, g!, ibeta, ConjugateGradient(),
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

# Default nonlinear procenditioner for `OACCEL`
# nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
#                            linesearch=LineSearches.Static())

# HagerZhang (Taken from the Conjugate Gradient implementation by Hager and Zhang, 2006)
# MoreThuente (From the algorithm in More and Thuente, 1994)
# BackTracking (Described in Nocedal and Wright, 2006)
# StrongWolfe (Nocedal and Wright)
# Static (Takes the proposed initial step length.)

# OACCEL(;manifold::Manifold = Flat(),
#        alphaguess = LineSearches.InitialStatic(),
#        linesearch = LineSearches.HagerZhang(),
#        nlprecon = GradientDescent(
#            alphaguess = LineSearches.InitialStatic(alpha=1e-4,scaled=true),
#            linesearch = LineSearches.Static(),
#            manifold = manifold),
#        nlpreconopts = Options(iterations = 1, allow_f_increases = true),
#        ϵ0 = 1e-12,
#        wmax::Int = 10)

nlprecon = ConjugateGradient(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
                             linesearch=LineSearches.Static())
# Default size of subspace that OACCEL accelerates over is `wmax = 10`
oacc10 = OACCEL(nlprecon=nlprecon, wmax=10)
oacc5 = OACCEL(nlprecon=nlprecon, wmax=5)
ngmres5 = NGMRES(nlprecon=nlprecon, wmax=5)

precongd = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
                           linesearch=LineSearches.Static())
gdoacc10 = OACCEL(nlprecon=precongd, wmax=10)
gdoacc5 = OACCEL(nlprecon=precongd, wmax=5)
gdngmres5 = NGMRES(nlprecon=precongd, wmax=5)

res12a = Optim.optimize(f, g!, ibeta, oacc10,
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

res12b = Optim.optimize(f, g!, ibeta, oacc5,
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

res12c = Optim.optimize(f, g!, ibeta, ngmres5,
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

res12d = Optim.optimize(f, g!, ibeta, gdoacc10,
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

res12e = Optim.optimize(f, g!, ibeta, gdoacc5,
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

res12f = Optim.optimize(f, g!, ibeta, gdngmres5,
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

precongd = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
  linesearch=LineSearches.Static())


precongd = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
  linesearch=LineSearches.StrongWolfe())
gdoacc5 = OACCEL(nlprecon=precongd, wmax=5)

res12e1 = Optim.optimize(f, g!, ibeta, gdoacc5,
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

res12e1 = Optim.optimize(f, g!, ibeta,
  OACCEL(nlprecon = Optim.GradientDescent(alphaguess = LineSearches.InitialStatic(alpha=1e-4,scaled=true),
                    linesearch = LineSearches.Static()),
       nlpreconopts = Optim.Options(iterations = 1, allow_f_increases = false),
       ϵ0 = 1e-12,
       wmax = 10),
  Optim.Options(g_tol = 1e-6, iterations = 100, store_trace = true, show_trace = true))

# OACCEL(;manifold::Manifold = Flat(),
#        alphaguess = LineSearches.InitialStatic(),
#        linesearch = LineSearches.HagerZhang(),
#        nlprecon = GradientDescent(
#            alphaguess = LineSearches.InitialStatic(alpha=1e-4,scaled=true),
#            linesearch = LineSearches.Static(),
#            manifold = manifold),
#        nlpreconopts = Options(iterations = 1, allow_f_increases = true),
#        ϵ0 = 1e-12,
#        wmax::Int = 10)


#  res12 100: 7.841757e+07
#  res12a 100: 8.462381e+07
# res12b 100: 8.536739e+07
# res12c 100: 9.110727e+07
# res12d 100: 7.717282e+07
# res12e 100: 7.574053e+07
# res12f 100: 8.530660e+07
# res12e1 100 gd, lsearch HZ: 7.585124e+07
# res12e1 100 gd, lsearch MT: 7.603634e+07
# res12e1 100 gd, lsearch BT: 7.574053e+07
# res12e1 100 gd, lsearch SW: 7.751540e+07

# Default nonlinear procenditioner for `OACCEL`
nlprecon = GradientDescent(alphaguess=LineSearches.InitialStatic(alpha=1e-4,scaled=true),
                           linesearch=LineSearches.Static())
# Default size of subspace that OACCEL accelerates over is `wmax = 10`
oacc10 = OACCEL(nlprecon=nlprecon, wmax=10)
Optim.optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, oacc10)

oacc5 = OACCEL(nlprecon=nlprecon, wmax=5)
Optim.optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, oacc5)

# now NGMRES
ngmres5 = NGMRES(nlprecon=nlprecon, wmax=5)
Optim.optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, ngmres5)



# %% nlboxsolve
using NLboxsolve

lb = fill(-Inf, length(ibeta))
ub = fill(Inf, length(ibeta))

Zygote.hessian(x -> sum(x.^3), [1 2; 3 4])
Zygote.hessian(sin, pi/2)

Zygote.hessian(f, ibeta)

ibeta = zeros(length(geotargets))
# soln_d = nlboxsolve(example!,x0,lb,ub,ftol=1e-15,xtol=1e-15,method=:lm)
@time sol = nlboxsolve(fvec, ibeta, method = :lm_ar, maxiters=30) # nice 388.999430 secs, f 76739.56030287837
@time sol = nlboxsolve(fvec, lsres.param, method = :lm_ar, maxiters=30)
@time sol = nlboxsolve(fvec, jfvec, ibeta, method = :lm_ar, maxiters=30)
@time sol = nlboxsolve(fvec, jfvec, sol.zero, method = :lm_ar, maxiters=30)
@time sol = nlboxsolve(fvec, ibeta, method = :newton) # fast, but memory? 379.192662 sec; produces Nans
@time sol = nlboxsolve(fvec, ibeta, method = :lm)
@time sol = nlboxsolve(fvec, ibeta, method = :lm_kyf)
@time sol = nlboxsolve(fvec, ibeta, method = :lm_fan)
@time sol = nlboxsolve(fvec, ibeta, method = :dogleg) # nice
@time sol = nlboxsolve(fvec, jfvec, ibeta, method = :dogleg) # nice
@time sol = nlboxsolve(fvec, ibeta, method = :nk) # nice 127.701281 sec, f 414160.287416278
@time sol = nlboxsolve(fvec, ibeta, lb, ub, method = :nk, xtol=1e-15, ftol=1e-15) # nice 128.206980 sec, f 414160.414160.287416278
@time sol = nlboxsolve(fvec, ibeta, method = :nk_fs, xtol=1e-15, ftol=1e-15, maxiters=10) # nice

@time sol = nlboxsolve(fvec2, ibeta, method = :nk)

function j(beta)
  Zygote.hessian(f, beta)
end
j(ibeta)

@time sol = nlboxsolve(fvec, j, ibeta, method = :lm_ar, maxiters=5)

ibetan = convert(Vector{Number}, ibetan)
@time sol = nlboxsolve(fvec2, ibetan, method = :jfnk, krylovdim=30) # error
# ERROR: MethodError: no method matching isless(::ComplexF64, ::ComplexF64)
# maxiters=100 xtol::T=1e-8,ftol::T=1e-8 krylovdim::S=30

sol.trace
sol

sol.iters
sol.trace
sol.zero
f(sol.zero)
sum(sol.fzero.^2) # residuals
f(ibeta)
f(lsres.param)

vals = fvec(ibeta)
typeof(vals)
typeof(ibeta)

function fvec2(beta)
  # beta = reshape(beta, size(geotargets))
  beta = convert(Vector{Number}, beta)
  res = psf.objvec(beta, wh, xmat, geotargets)
  convert(Vector{Number}, res)
end
vals = fvec2(ibeta)
typeof(vals)
typeof(ibeta)


sol.iters
sol.trace
sol.zero
f(sol.zero)
sum(sol.fzero.^2) # residuals

# constrained Newton (method = :newton)
# constrained Levenberg-Marquardt (method = :lm)
# Kanzow, Yamashita, and Fukushima (2004) (method = :lm_kyf)
# Fan (2013) (method = :lm_fan)
# Amini and Rostami (2016) (method = :lm_ar) (this is the default method)
# Bellavia, Macconi, and Pieraccini (2012) (method = :dogleg)
# Chen and Vuik (2016) (method = :nk)
# a constrained globalized Newton-Krylov method based on Frontini and Sormani (2004) (method = :nk_fs)
# Jacobian-free Newton Krylov (method = :jfnk)



sol = nlboxsolve(fvec, ibeta, xtol=1e-15, ftol=1e-15)
nlboxsolve(fvec, ibeta, method=:jfnk)
nlboxsolve(fvec, ibeta, xtol=1e-15, ftol=1e-15, method=:jfnk)
nlboxsolve(fvec, ibeta, xtol=1e-15, ftol=1e-15, krylovdim=80, method=:jfnk)

fvec(ibeta)

function fivediagonal(x)

    f = similar(x)

    f[1]     = 4.0*(x[1] - x[2]^2) + x[2] - x[3]^2
    f[2]     = 8.0*x[2]*(x[2]^2 - x[1]) - 2.0*(1.0 - x[2]) + 4.0*(x[2] - x[3]^2) + x[3] - x[4]^2
    f[end-1] = 8.0*x[end-1]*(x[end-1]^2 - x[end-2]) - 2.0*(1.0 - x[end-1]) + 4.0*(x[end-1] - x[end]^2)
             + x[end-2]^2 - x[end-3]
    f[end]   = 8.0*x[end]*(x[end]^2 - x[end-1]) - 2*(1.0 - x[end]) + x[end-1]^2 - x[end-2]
    for i = 3:length(x)-2
        f[i] = 8.0*x[i]*(x[i]^2 - x[i-1]) - 2.0*(1.0 - x[i]) + 4.0*(x[i] - x[i+1]^2) + x[i-1]^2
             - x[i-2] + x[i+1] - x[i+2]^2
    end

    return f

end

function fivediagonal!(f,x)

    f[1]     = 4.0*(x[1] - x[2]^2) + x[2] - x[3]^2
    f[2]     = 8.0*x[2]*(x[2]^2 - x[1]) - 2.0*(1.0 - x[2]) + 4.0*(x[2] - x[3]^2) + x[3] - x[4]^2
    f[end-1] = 8.0*x[end-1]*(x[end-1]^2 - x[end-2]) - 2.0*(1.0 - x[end-1]) + 4.0*(x[end-1] - x[end]^2)
             + x[end-2]^2 - x[end-3]
    f[end]   = 8.0*x[end]*(x[end]^2 - x[end-1]) - 2*(1.0 - x[end]) + x[end-1]^2 - x[end-2]
    for i = 3:length(x)-2
        f[i] = 8.0*x[i]*(x[i]^2 - x[i-1]) - 2.0*(1.0 - x[i]) + 4.0*(x[i] - x[i+1]^2) + x[i-1]^2
            - x[i-2] + x[i+1] - x[i+2]^2
    end

end

n = 5000
x0 = [2.0 for _ in 1:n]
@time soln_a = nlboxsolve(fivediagonal,x0,xtol=1e-15,ftol=1e-15,krylovdim=80,method=:jfnk)
@time soln_b = nlboxsolve(fivediagonal!,x0,xtol=1e-15,ftol=1e-15,krylovdim=80,method=:jfnk)


# %% OptimPack and OptimPackNextGen
# https://github.com/emmt/OptimPackNextGen.jl
# https://github.com/emmt/OptimPackNextGen.jl/blob/master/doc/quasinewton.md


# add https://github.com/emmt/OptimPack.jl

using OptimPack

# The easiest way to use these minimizers is to provide a Julia function, say fg!, which is in charge of computing the value of the function and its gradient
# for given variables. This function must have the form:

# function fg!(x, g)
#   g[...] = ... # store the gradient of the function
#   f = ...      # compute the function value
#   return f     # return the function value
# end

function fg!(beta, g)
  g = Zygote.gradient(f, beta) # store the gradient of the function
  return f(beta)     # return the function value
end

method = OptimPack.NLCG_POLAK_RIBIERE_POLYAK | OptimPack.NLCG_POWELL

x = nlcg(fg!, ibeta, method)

x2 = vmlm(fg!, ibeta, m=3)

x = vmlmb(fg!, x0; mem=..., lower=..., upper=..., ftol=..., fmin=...)


# %% newton
# https://discourse.julialang.org/t/optim-jl-oncedifferentiable-baby-steps/10185
# https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/maxlikenlm/
# max likelihood example
n = 500                             # Number of observations
nvar = 2                            # Number of variables
β = ones(nvar) * 3.0                # True coefficients
x = [ones(n) randn(n, nvar - 1)]    # X matrix of explanatory variables plus constant
ε = randn(n) * 0.5                  # Error variance
y = x * β + ε;                      # Generate Data

function Log_Likelihood(X, Y, β, log_σ)
  σ = exp(log_σ)
  llike = -n/2*log(2π) - n/2* log(σ^2) - (sum((Y - X * β).^2) / (2σ^2))
  llike = -llike
end

func = TwiceDifferentiable(vars -> Log_Likelihood(x, y, vars[1:nvar], vars[nvar + 1]),
                           ones(nvar+1); autodiff=:forward);

opt = optimize(func, ones(nvar+1))

# %% newton my data
# https://discourse.julialang.org/t/is-there-a-julia-package-that-uses-newton-krylov-method-for-systems-of-nonlinear-equations/36520/5
fnewt = TwiceDifferentiable(f, ibeta; autodiff=:forward)
resnewt = optimize(fnewt, ibeta, Optim.Options(iterations = 100, store_trace = true, show_trace = true))

td = TwiceDifferentiable(f, initial_x; autodiff = :forward)
Optim.minimizer(optimize(td, initial_x, Newton()))


# %% hessians
# https://gitter.im/JuliaNLSolvers/Optim.jl?at=5f43ca8bec534f584fbaf88b
using ModelingToolkit
using Optim
@variables x[1:2]
f1 = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
g1 = ModelingToolkit.gradient(f1, x)
h1 = ModelingToolkit.hessian(f1, x)

buildedF = build_function(f1, x[1:2])
buildedG = build_function(g1, x[1:2])
buildedH = build_function(h1, x[1:2])

newF = eval(buildedF)
newG! = eval(buildedG[2])
newH! = eval(buildedH[2])

initial_x = zeros(2)

@time Optim.minimizer(optimize(newF, initial_x, Newton(); autodiff=:forward))
@time Optim.minimizer(optimize(newF, newG!, newH!, initial_x, Newton()))

# %% GalacticOptim
# https://discourse.julialang.org/t/best-nonlinear-optimizer-for-a-continuous-convex-problem/77771/11
# https://docs.juliahub.com/GalacticOptim/fP6Iz/3.4.0/tutorials/intro/
# import Pkg; Pkg.add("Optimization")

# this may work
using GalacticOptim
using GalacticOptimJL
using Zygote
using Optim
# rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
fn(x) = sum(x.^2)
# x0 = zeros(5)
x0 = [1., 2.0, 3.0, 4.0, 5.0]
fn(x0)
# p  = [1.0,100.0]
# using ForwardDiff
fng = GalacticOptim.OptimizationFunction(fn, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(fng, x0)

sol = GalacticOptim.solve(prob, Optim.KrylovTrustRegion())


# again? - this seems to do it -- having the dummy argument here called p
using GalacticOptim
using GalacticOptimJL
using Zygote
using Optim
# rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
fn(x, p) = sum(x.^2)
# x0 = zeros(5)
x0 = [1., 2.0, 3.0, 4.0, 5.0]
fn(x0, nothing)
# p  = [1.0,100.0]
# using ForwardDiff
fn2 = GalacticOptim.OptimizationFunction(fn, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(fn2, x0, nothing)
# using Optimization
sol = GalacticOptim.solve(prob, Optim.KrylovTrustRegion())






using GalacticOptim
rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = OptimizationProblem(rosenbrock,x0,p)

using GalacticOptimJL
sol = solve(prob,NelderMead())

using GalacticBBO
prob = OptimizationProblem(rosenbrock, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited())

sol.original

using ForwardDiff
f3 = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(f3, x0, p)
sol = solve(prob,BFGS())

prob = OptimizationProblem(f3, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, Fminbox(GradientDescent()))

sol

# %% least squares
LeastSquaresOptim.optimize(f, ibeta, Dogleg())
f(ibeta)

LeastSquaresOptim.optimize!(LeastSquaresProblem(x = ibeta,
                                f! = f, output_length = 2))


# %% ipopt
using JuMP, Ipopt
m = Model(Ipopt.Optimizer)

@variable(m, x[1:2])
@NLobjective(m, Min, (x[1]-3)^3 + (x[2]-4)^2)
@NLconstraint(m, (x[1]-1)^2 + (x[2]+1)^3 + exp(-x[1]) <= 1)

JuMP.optimize!(m)
println("** Optimal objective function value = ", JuMP.objective_value(m))
println("** Optimal solution = ", JuMP.value.(x))

include("nlp_ipopt.jl")



# %% krylov trust -- worth more investigation
f2(ibeta, nothing)
# p  = [1.0,100.0]
# using ForwardDiff
fn2 = GalacticOptim.OptimizationFunction(f2, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(fn2, ibeta, nothing)
prob3 = GalacticOptim.OptimizationProblem(fn2, minimizer(res13a), nothing)
prob3 = GalacticOptim.OptimizationProblem(fn2, sol6a.minimizer, nothing)
prob3 = GalacticOptim.OptimizationProblem(fn2, results.zero, nothing)
# using Optimization
sol = GalacticOptim.solve(prob3, Optim.KrylovTrustRegion(), maxiters = 1_000, store_trace = true, show_trace = true, show_every=10)

sol = GalacticOptim.solve(prob3, Optim.NewtonTrustRegion(), maxiters = 1_000, store_trace = true, show_trace = true, show_every=1)

@time sol = GalacticOptim.solve(prob, Optim.NewtonTrustRegion(), maxiters = 20, store_trace = true)  # 2360 secs, obj 61.9m
@time sol = GalacticOptim.solve(prob, Optim.Newton(), maxiters = 20, store_trace = true)
@time sol = GalacticOptim.solve(prob, Optim.NGMRES(), maxiters = 200, store_trace = true, show_trace = true, show_every=20)
@time sol = GalacticOptim.solve(prob, Optim.OACCEL(), maxiters = 20000, store_trace = true, show_trace = true, show_every=40)
# Optim.NGMRES()
# Optim.OACCEL()

sol.minimum
f(sol.minimizer)

# sol2 = solve(prob, Fminbox(GradientDescent()), show_trace = true)
# sol2 = solve(prob, Optim.Fminbox(GradientDescent()), show_trace = true, maxiters=3)

sol3 = solve(prob, Newton(), show_trace = true, maxiters=3)
sol3.minimum
sol3.minimizer

@time sol3a = solve(prob3, Newton(), show_trace = true, maxiters=3)
sol3a.minimum

@time sol3b = solve(prob3, NewtonTrustRegion(), show_trace = true, maxiters=3)
sol3b.minimum # 1.1979870851977023e8 barely moved from ibeta

sol4 = solve(prob, LBFGS(), show_trace = true, maxiters=100)
sol4.minimum
# sol5 = solve(prob, ADAM(0.1), maxiters = 10, show_trace = true)

using GalacticMOI
using Ipopt
sol5 = solve(prob, Ipopt.Optimizer(), maxiters=2) # takes TOO LONG

using GalacticNLopt
# NLopt gradient-based optimizers are:
# NLopt.LD_LBFGS_NOCEDAL() NLopt.LD_LBFGS() NLopt.LD_VAR1() NLopt.LD_VAR2()
# NLopt.LD_TNEWTON() NLopt.LD_TNEWTON_RESTART() NLopt.LD_TNEWTON_PRECOND() NLopt.LD_TNEWTON_PRECOND_RESTART()
# NLopt.LD_MMA() NLopt.LD_AUGLAG() NLopt.LD_AUGLAG_EQ() NLopt.LD_SLSQP() NLopt.LD_CCSAQ()
# sol = solve(prob, Opt(:LN_BOBYQA, 2))
# sol6 = solve(prob, Opt(:LD_SLSQP))
sol6 = solve(prob, NLopt.LD_LBFGS(), maxiters=100)
sol6 = solve(prob, NLopt.LD_MMA(), maxiters=1_000, show_trace = true)
sol6.minimum

callback2 = function(p, l)
  # println(iter, l)
  # println("++++++++++++++++++")
  println(l)

  # global iter
  # iter += 1

  return false
end
@time sol6 = solve(prob, NLopt.LD_MMA(), callback=callback2, maxiters=10_000)
sol6.minimum

prob2 = GalacticOptim.OptimizationProblem(fn2, sol6.minimizer, nothing)
@time sol6a = solve(prob2, NLopt.LD_MMA(), maxiters=10_000, callback=callback2)
sol6a.minimum
# at 100
# LD_LBFGS 1.4531892038182566e7
# LD_VAR1 2.814110739921908e7
# LD_VAR2 same
# LD_MMA at 100 511749.0600063354; at 1_000 6970.619922676372; at 10_000 3201.2890169654047, at 16_000 2495.509254966556
#   at 21_000 1616.3093901287261; 26k 1180.2242696991439; 31k ; 36k
# LD_SLSQP 1.5771092439317225e7
# LD_CCSAQ 1.3863958717007548e7
# LD_TNEWTON 9.370676362182084e6
# LD_TNEWTON_RESTART 1.914081009433879e7
# LD_TNEWTON_PRECOND 1.727017879051921e7
# LD_TNEWTON_PRECOND_RESTART 5.195132775119288e7
#
@time sol_temp = solve(prob, NLopt.LD_MMA(), maxiters=10)

# result1 = GalacticOptim.solve(optprob1, ADAM(0.01), callback=callback, maxiters=10)
# https://discourse.julialang.org/t/galacticoptim-solve-ignores-callback/81670
callback = function(p, l)
  println("++++++++++++++++++")
  # println(p, l)
  println(l)
  return false
end
sol_temp = solve(prob, NLopt.LD_MMA(), maxiters=10, callback=callback)



sol.minimum
sol.minimizer - sol.u

beta_o = reshape(sol6a.minimizer, size(geotargets))


# https://galacticoptim.sciml.ai/dev/API/optimization_problem/
# https://galacticoptim.sciml.ai/stable/API/optimization_problem/
using GalacticOptim
using GalacticOptimJL
using ModelingToolkit
prob = OptimizationProblem(f, ibeta, p=nothing)
OptimizationProblem(f, x, p = DiffEqBase.NullParameters(),;
                    lb = nothing,
                    ub = nothing,
                    lcons = nothing,
                    ucons = nothing,
                    kwargs...)
# OptimizationFunction(f, AutoModelingToolkit(), x0,p,
#                      grad = false, hess = false, sparse = false,
#                      checkbounds = false,
#                      linenumbers = true,
#                      parallel=SerialForm(),
#                      kwargs...)
OptimizationFunction(f, AutoZygote())

prob = OptimizationProblem(OptimizationFunction(f, GalacticOptim.AutoZygote()), ibeta)
# ERROR: MethodError: no method matching f(::Vector{Float64}, ::SciMLBase.NullParameters)

function f2(beta::Vector{Float64}, parms::SciMLBase.NullParameters)
  # beta = reshape(beta, size(geotargets))
  psf.objfn(beta, wh, xmat, geotargets) / obj_scale
end

prob = OptimizationProblem(OptimizationFunction(f2, GalacticOptim.AutoZygote()), ibeta)

# OptimizationFunction(f, AutoModelingToolkit(), ibeta,
#                      grad = true, hess = false, sparse = false,
#                      checkbounds = false,
#                      linenumbers = true)
sol = solve(prob, Optim.KrylovTrustRegion())

using GalacticOptim
abc = OptimizationFunction(f; grad = GalacticOptim.AutoForwardDiff(), hes = GalacticOptim.AutoForwardDiff())

prob = OptimizationProblem(OptimizationFunction(f, GalacticOptim.AutoZygote()), ibeta)
prob = OptimizationProblem(
  OptimizationFunction(f; grad = GalacticOptim.AutoForwardDiff(), hes = GalacticOptim.AutoForwardDiff()),
  ibeta)
solve(prob, Optim.KrylovTrustRegion())

# ERROR: Use OptimizationFunction to pass the derivatives or automatically generate them with one of the autodiff backends

# 2.005618e+07 # .4 default
# 1.368090e+07 # .01
# 1.315721e+07 # .001
# 1.127418e+07 # 0
# 1.236774e+07 # 1.0

# %% roots nlsolve ========================
using NLsolve
f2(x) = [(x[1]+3)*(x[2]^3-7)+18
        sin(x[2]*exp(x[1])-1)] # returns an array

results = nlsolve(f2, [ 0.1; 1.2])
results = nlsolve(f2, [ 0.1; 1.2], autodiff=:forward)

function pd2(beta, wh, xmat, geotargets)
  beta = reshape(beta, size(geotargets))
  whs = psf.geo_weights(beta, wh, xmat)
  calctargets = psf.geo_targets(whs, xmat)
  pdiffs = psf.targ_pdiffs(calctargets, geotargets)
  pdiffs
end

function fp2(beta)
  pd2(beta, wh, xmat, geotargets)
end

fp2(ibeta)

# method = :trust_region

# method = :newton
# This method accepts a custom parameter linesearch, which must be equal to a function computing the linesearch. Currently, available values are taken from
# the LineSearches package. By default, no linesearch is performed. Note: it is assumed that a passed linesearch function will at least update the solution vector and evaluate the function at the new point.
# HagerZhang (Taken from the Conjugate Gradient implementation by Hager and Zhang, 2006)
# MoreThuente (From the algorithm in More and Thuente, 1994)
# BackTracking (Described in Nocedal and Wright, 2006)
# StrongWolfe (Nocedal and Wright)
# Static (Takes the proposed initial step length.)


# method = :anderson ERROR: LAPACKException(1)
# linesearches
@time results = nlsolve(fvec, ibeta, autodiff=:forward, method = :newton, linesearch=LineSearches.Static(), iterations=5, show_trace = true)
# BAD @time results = nlsolve(fvec, ibeta, autodiff=:forward, method = :newton, linesearch=LineSearches.StrongWolfe(), iterations=5, show_trace = true) # seems to get stuck
@time results = nlsolve(fvec, ibeta, autodiff=:forward, method = :newton, linesearch=LineSearches.BackTracking(), iterations=5, show_trace = true)
@time results = nlsolve(fvec, ibeta, autodiff=:forward, method = :newton, linesearch=LineSearches.MoreThuente(), iterations=5, show_trace = true)

@time results = nlsolve(fvec, ibeta, autodiff=:forward, method = :broyden, linesearch=LineSearches.HagerZhang(), iterations=50, show_trace = true)

@time results = nlsolve(fvec, ibeta, autodiff=:forward, method = :anderson, m=100, iterations=1000, show_trace = true)


@time results = nlsolve(fvec, ibeta, autodiff=:forward, autoscale=false, method = :trust_region, factor=5.0, iterations=20, show_trace = true)
# @time results = nlsolve(fvec, ibeta, autodiff=:forward, show_trace = true)
f(results.zero)
f(ibeta)

# lsres 20 iter, 92 secs, was 154,3305
# 20 iters, around 380 secs
# factor 10  672,810.8733569914
# factor 5.0 192,098.57983745285 (7.8415e01); with autoscale=false, 1.9e7
# factor 2.0 1.1e6
# factor 4.0 973,860
# factor 7.5 344.143
# factor 6.0 307,944.4451871131

@time results2 = nlsolve(fvec, results.zero, autodiff=:forward, autoscale=true, method = :trust_region, factor=1.0, iterations=20, show_trace = true)
@time results2 = nlsolve(fvec, results.zero, autodiff=:forward, autoscale=true, method = :trust_region, factor=1.0, iterations=20, autoscale=false, show_trace = true)

stop

# %% stop

# prob9 287.276564, f 0

# stub 2:
# julia> @time results = nlsolve(fvec, ibeta, autodiff=:forward, autoscale=true, method = :trust_region, iterations=50, show_trace = true)
# 89352.32683331153
# Iter     f(x) inf-norm    Step 2-norm
# ------   --------------   --------------
#      0     3.170699e+03              NaN
#      1     3.167727e+03     2.375773e+00
#      2     3.161785e+03     4.752105e+00
#      3     3.149909e+03     9.506449e+00
#      4     3.126197e+03     1.902189e+01
#      5     3.078967e+03     3.807994e+01
#      6     2.985563e+03     7.630443e+01
#      7     2.804467e+03     1.531699e+02
#      8     2.466706e+03     3.084341e+02
#      9     2.211088e+03     6.259345e+02
#     10     1.745168e+03     1.299258e+03
#     11     8.253235e+02     2.713203e+03
#     12     3.031735e+02     3.165599e+03
#     13     5.885888e+02     2.705933e+03
#     14     5.027225e+02     5.025450e+03
#     15     5.683660e+02     5.042770e+03
#     16     8.489918e+02     3.952950e-13
#     17     8.347716e+02     2.849997e-13
#     18     8.277964e+02     5.665081e-13
#     19     8.243288e+02     2.972180e-13
#     20     8.225844e+02     3.485650e-13
#     21     8.216763e+02     5.443656e-14
#     22     8.210907e+02     2.298425e-13
#     23     4.504602e+02     1.744964e+02
#     24     6.199373e+02     1.743914e-13
#     25     5.510673e+02     5.913161e-14
#     26     1.519673e+02     9.635946e+01
#     27     1.251351e+02     4.458129e+02
#     28     1.737148e+02     1.367429e+03
#     29     1.111398e+02     2.917128e+03
#     30     1.653617e+02     1.682411e+03
#     31     9.251482e+01     4.687297e+03
#     32     3.506633e+02     2.071615e-12
#     33     8.689092e+01     1.919204e+04
#     34     4.584845e+02     7.629692e-13
#     35     1.357286e+02     4.069544e+03
#     36     1.310267e+02     2.024218e+04
#     37     8.374561e+01     3.984989e+04
#     38     1.108168e+02     4.475483e+04
#     39     1.203225e+02     1.200658e+05
#     40     5.373188e+01     1.388321e+05
#     41     5.321076e+01     1.318710e+05
#     42     5.309995e+01     1.073503e+06
#     43     5.280050e+01     6.560022e+05
#     44     5.265706e+01     2.258855e+06
#     45     5.205422e+01     4.734785e+06
#     46     5.191688e+01     1.164533e+07
#     47     5.166083e+01     9.866703e+06
#     48     5.155104e+01     2.298473e+07
#     49     5.138325e+01     6.346780e+07
#     50     5.145441e+01     2.110034e+08
# 741.791642 seconds (149.46 k allocations: 2.053 TiB, 11.42% gc time)








# %% OLD reverse differentiation
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
# https://julia.quantecon.org/more_julia/optimization_solver_packages.html
# https://www.youtube.com/watch?v=Sv3d0k7wWHk

using Zygote

using Optim, LinearAlgebra
N = 1000000
y = rand(N)
λ = 0.01
obj(x) = sum((x .- y).^2) + λ*norm(x)

x_iv = rand(N)
function g!(G, x)
    G .=  obj'(x)
end

results = optimize(obj, g!, x_iv, LBFGS()) # or ConjugateGradient()
println("minimum = $(results.minimum) with in "*
"$(results.iterations) iterations")

results2 = optimize(obj, x_iv, LBFGS(); autodiff = :forward)

#   https://discourse.julialang.org/t/strange-errors-occurred-when-i-optimized-with-zygote-and-optim/47461
using Zygote
using Optim
using RecursiveArrayTools
#This is the target function that I’m optimizing
function get_f(A)
  M=reshape(1:16,4,4)
  U=exp(A)
  T=UMU’
  return tr(T)
end
function my_function()
  g(A)=gradient(A->get_f(A), A)[1] # This is my gradient function
  A0=zeros(4,4)
  results=optimize(get_f,g,VectorOfArray([A0]), LBFGS())
end

Zygote.gradient(x -> 3x^2 + 2x + 1, 5)

function f1(x)
    3x^2 + 2x + 1
end

f1(3)

g = Zygote.gradient(f1, 5)
g
typeof(g)

Zygote.gradient(f1, x)

Zygote.gradient(f1)
gradient(model -> sum(model(x)), model)

# %% misc notes


# %% end


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
