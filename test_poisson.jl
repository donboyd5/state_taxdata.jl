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
using Tables
using MINPACK
using LsqFit
using LeastSquaresOptim
using GalacticOptim
using GalacticOptimJL

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
    # beta = reshape(beta, size(geotargets))
    psf.objfn(beta, wh, xmat, geotargets) / obj_scale
end

function g!(G, x)
  G .=  f'(x)
end

function f2(beta, dummy)
  # dummy added for GalacticOptim
  psf.objfn(beta, wh, xmat, geotargets) / obj_scale
end

function fvec(beta)
  # beta = reshape(beta, size(geotargets))
  psf.objvec(beta, wh, xmat, geotargets)
end

function fvec!(out, beta)
  # for LeastSquaresOptim inplace
  out .= psf.objvec(beta, wh, xmat, geotargets)
end


# %% get a problem

tp = mtp.get_rproblem()

tp = mtp.mtp(10, 3, 2)
tp = mtp.mtp(100, 8, 4)
tp = mtp.mtp(1000, 12, 6)
tp = mtp.mtp(10000, 25, 12)
tp = mtp.mtp(50000, 50, 20)
tp = mtp.mtp(100000, 75, 40)

tp = gtp.get_taxprob(1)


# %% unpack the problem
h = size(tp.xmat)[1]
# s = size(tp.geotargets)[1]
# k = size(tp.geotargets)[2]

# get data
xmat = tp.xmat
wh = tp.wh
geotargets = tp.geotargets


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

# %% examine if desired
nattargs = sum(geotargets, dims=1) # national value for each target
sum(wh)

calcnat = sum(wh .* xmat, dims=1) # national value for each target at national weights
calcdiffs = calcnat .- nattargs
calcpdiffs = calcdiffs ./ nattargs
quantile(vec(calcpdiffs))


# %% levenberg marquardt
# https://github.com/sglyon/MINPACK.jl

# https://julianlsolvers.github.io/LsqFit.jl/latest/
# https://github.com/JuliaNLSolvers/LsqFit.jl/
# see Geodesic acceleration -- can I use curve_fit and this approach?
# re Geodesic acceleration see https://www.gnu.org/software/gsl/doc/html/nls.html

# also examine:
# https://juliapackages.com/p/leastsquaresoptim #  written with large scale problems in mind
# https://github.com/matthieugomez/LeastSquaresOptim.jl

# and possibly https://github.com/JuliaNLSolvers/NLsolve.jl

ibeta = zeros(length(geotargets))
# ibeta = randn(length(tp.geotargets))

# LsqFit.lmfit(cost_function, [0.5, 0.5], Float64[])
fvec(ibeta)
f(ibeta)
# @time lsres = LsqFit.lmfit(fvec, ibeta, Float64[]; show_trace=true, maxIter=3)
# # 194.759903 secs

@time lsres = LsqFit.lmfit(fvec, ibeta, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=5)
# @time lsres = LsqFit.lmfit(fvec, lsres.param, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=100)
# 68.466341 secs for 3 iterations, 313.606478 secs for 30 iterations

# @time lsres3 = LsqFit.lmfit(fvec, lsres3.param, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=10)

# prob9 21 iter 225.801845 seconds

f(lsres.param)

# fieldnames(lsres)
lsres.param
# LsqFit.lmfit(cost_function, [0.5, 0.5], Float64[], autodiff=:forward)
# LsqFit.LsqFitResult{Array{Float64,1},Array{Float64,1},Array{Float64,2},Array{Float64,1}}([2.0, 0.5], [4.44089e-16, 8.88178e-16, 1.77636e-15, 1.77636e-15, 1.77636e-15, 3.55271e-15, 3.55271e-15, 3.55271e-15, 3.55271e-15, 3.55271e-15], [-1.0 -0.0; -2.0 -0.0; … ; -9.0 -0.0; -10.0 -0.0], true, Float64[])

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
@. model(x, p) = p[1]*exp(-x*p[2])

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



# %% examine results

# dump(res)
res = res12
beta_o = reshape(minimizer(res), size(geotargets))

# levenberg marquardt results
res = lsres.param
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
# For dense jacobians, the default option is Doglel(QR()).
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

# @time lsores = LeastSquaresOptim.optimize(fvec, ibeta, Dogleg(LeastSquaresOptim.QR()),
#  autodiff = :forward, show_trace=true, iterations=30)

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
res5 = Optim.optimize(f, ibeta, ConjugateGradient(),
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
  Optim.Options(g_tol = 1e-6, iterations = 10_000, store_trace = true, show_trace = true))
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

res13a = Optim.optimize(f, g!, minimizer(res17a), GradientDescent(),
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

fnewt = TwiceDifferentiable(f, ibeta; autodiff=:forward)

resnewt = optimize(fnewt, ibeta, Optim.Options(iterations = 100, store_trace = true, show_trace = true))

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



# %% krylov trust
f2(ibeta, nothing)
# p  = [1.0,100.0]
# using ForwardDiff
fn2 = GalacticOptim.OptimizationFunction(f2, GalacticOptim.AutoZygote())
prob = GalacticOptim.OptimizationProblem(fn2, ibeta, nothing)
prob3 = GalacticOptim.OptimizationProblem(fn2, minimizer(res13a), nothing)
prob3 = GalacticOptim.OptimizationProblem(fn2, sol6a.minimizer, nothing)
# using Optimization
sol = GalacticOptim.solve(prob3, Optim.KrylovTrustRegion(), maxiters = 10_000, store_trace = true, show_trace = true, show_every=10)

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
sol6 = solve(prob, NLopt.LD_MMA(), maxiters=10_000)
sol6.minimum

callback2 = function(p, l)
  # println(iter, l)
  # println("++++++++++++++++++")
  println(l)

  # global iter
  # iter += 1

  return false
end
@time sol6 = solve(prob, NLopt.LD_MMA(), callback=callback2, maxiters=1_000)
sol6.minimum


prob2 = GalacticOptim.OptimizationProblem(fn2, sol6a.minimizer, nothing)

@time sol6a = solve(prob2, NLopt.LD_MMA(), maxiters=5_000, callback=callback2)
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

# %% roots
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
results = nlsolve(fp2, ibeta)


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
