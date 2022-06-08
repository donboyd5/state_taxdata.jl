# GalacticOptim
# https://docs.juliahub.com/GalacticOptim/fP6Iz/3.4.0/


# Optimization
# https://github.com/SciML/Optimization.jl

# , "OptimizationSpeedMapping"
import Pkg; Pkg.add(["Optimization", "OptimizationOptimJL", "OptimizationBBO", "SpeedMapping"])
Pkg.add("OptimizationFlux")

using Optimization
rosenbrock(x,p) =  (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
p  = [1.0,100.0]

prob = Optimization.OptimizationProblem(rosenbrock,x0,p)

using OptimizationOptimJL
sol = Optimization.solve(prob,NelderMead())


using OptimizationBBO
prob = Optimization.OptimizationProblem(rosenbrock, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = Optimization.solve(prob,BBO_adaptive_de_rand_1_bin_radiuslimited())

using ForwardDiff
fn2 = Optimization.OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())
prob = Optimization.OptimizationProblem(fn2, x0, p)
sol = Optimization.solve(prob,BFGS())

prob = Optimization.OptimizationProblem(fn2, x0, p, lb = [-1.0,-1.0], ub = [1.0,1.0])
sol = Optimization.solve(prob, Fminbox(GradientDescent()))




using SpeedMapping
rosenbrock(x) =  (1 - x[1])^2 + 100(x[2] - x[1]^2)^2;
sol = speedmapping(zeros(2); f = rosenbrock).minimizer

sol = speedmapping(zeros(2); f = rosenbrock).minimizer


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



# %% get test problem
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

sol = speedmapping(ibeta; f = f)
dump(sol)
sol.minimizer
f(sol.minimizer)

# %% flux
using OptimizationFlux

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
x0 = zeros(2)
_p = [1.0, 100.0]
l1 = rosenbrock(x0, _p)

optprob = OptimizationFunction(rosenbrock, Optimization.AutoForwardDiff())

prob = OptimizationProblem(optprob, x0, _p)

sol = Optimization.solve(prob, Flux.ADAM(0.1), maxiters = 1000)
10 * sol.minimum < l1

prob = OptimizationProblem(optprob, x0, _p)
sol = solve(prob, Flux.ADAM(), maxiters = 1000, progress = true)

# %% my problem
function f2(beta, pdummy)
    # pdummy added for GalacticOptim
    psf.objfn(beta, wh, xmat, geotargets)*pdummy
end

function f2(beta)
    psf.objfn(beta, wh, xmat, geotargets)
end

prob1 = Optimization.OptimizationFunction(f2, Optimization.AutoForwardDiff())
pdummy = 1.0
prob2 = Optimization.OptimizationProblem(prob1, ibeta, pdummy)
f2(ibeta)
sol = Optimization.solve(prob2, Flux.RADAM(0.5), maxiters = 1000, progress = true)
sol.minimum
sol.minimizer