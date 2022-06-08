# test flux

# https://github.com/SciML/Optimization.jl

# %% libraries

import Pkg; Pkg.add("GalacticFlux")

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

using GalacticOptim, GalacticOptimJL, GalacticFlux


# %% load local modules
include("../jl/poisson_functions.jl")
import .poissonFunctions as psf

include("../jl/make_test_problems.jl")
import .makeTestProblems as mtp

include("../jl/getdata_functions.jl")
import .getTaxProblem as gtp

# include("jl/rproblem.jl")

# %% functions

function f(beta)
    psf.objfn(beta, wh, xmat, geotargets) / obj_scale
end

function g!(G, x)
  G .=  f'(x)
end

function f2(beta, pdummy)
  # pdummy added for GalacticOptim
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

function jfvec(beta)
  ForwardDiff.jacobian(x -> fvec(x), beta)
end

# %% examples
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

using ForwardDiff
fn = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(fn, x0, p)
sol = solve(prob, BFGS())
sol.minimum

prob = OptimizationProblem(fn, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = solve(prob, Fminbox(GradientDescent()))


using GalacticFlux
fn = OptimizationFunction(rosenbrock, GalacticOptim.AutoForwardDiff())
prob = OptimizationProblem(fn, x0, p)
# prob = OptimizationProblem(rosenbrock,x0,p)
sol = solve(prob, GalacticFlux.Descent(), maxiters=10_000)
# fieldnames(sol)
dump(sol)
sol.alg
sol.alg.eta
sol.minimum
sol.minimizer

sol = solve(prob, ADAM(), maxiters=10_000)
sol.minimum

# %% get test
tp = mtp.mtp(10, 3, 2)
tp = mtp.mtp(100, 8, 4)
tp = mtp.mtp(1000, 12, 6)
tp = mtp.mtp(5000, 20, 8)
tp = mtp.mtp(10000, 25, 12)
tp = mtp.mtp(50000, 50, 20)
tp = mtp.mtp(100000, 75, 40)

# %% unpack
h = size(tp.xmat)[1]
# s = size(tp.geotargets)[1]
# k = size(tp.geotargets)[2]

# get data
xmat = tp.xmat
wh = tp.wh
geotargets = tp.geotargets
ibeta = zeros(length(geotargets))

# %% prob
fn2 = OptimizationFunction(f, GalacticOptim.AutoForwardDiff())
prob2 = OptimizationProblem(fn2, ibeta)
f(ibeta)

sol = solve(prob2, ADAM(), maxiters=10)
sol.minimum

