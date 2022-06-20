

# https://ui.adsabs.harvard.edu/abs/2015AGUFM.H41B1299L/abstract
# https://mads.lanl.gov/#downloads
# https://github.com/madsjulia/Mads.jl
# https://gitlab.com/mads/Mads.jl


import Pkg; Pkg.add(Pkg.PackageSpec(name="Mads", rev="master"))

import Mads
import LinearAlgebra
Mads.test()
Mads.help()

# include(Mads.dir * "/../examples/contamination/contamination.jl")
include(Mads.dir * "/examples/contamination/contamination.jl")
include(Mads.dir * "/examples/MadsExamples.jl")

include(Mads.dir * "/examples/contamination/optimization-lm.jl")

include(Mads.dir * "/../examples/contamination/contamination.jl")

MadsLevenbergMarquardt.jl

@Mads.stderrcapture fopt(x) = [x[1], 2.0 - x[2]]
@Mads.stderrcapture gopt(x) = [1.0 0.0; 0.0 -1.0]

@time res1 = Mads.naive_levenberg_marquardt(fopt, gopt, [100.0, 100.0])
@Test.test LinearAlgebra.norm(res1.minimizer - [0.0, 2.0]) < 0.01

@time res2 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], show_trace=true)
@Test.test LinearAlgebra.norm(res2.minimizer - [0.0, 2.0]) < 0.01

@time res3 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], tolOF=1e-24)
@time res4 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], np_lambda=4, tolX=1e-8, tolG=1e-12)
@time res5 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], tolOF=1e-24, tolX=1e-24, tolG=1e-24)

sum(fopt(res1.minimizer).^2)
sum(fopt(res2.minimizer).^2)
sum(fopt(res3.minimizer).^2)
sum(fopt(res4.minimizer).^2)
sum(fopt(res5.minimizer).^2)

res1.minimum
res2.minimum
res3.minimum
res4.minimum
res5.minimum

#

res = Mads.levenberg_marquardt(Mads.rosenbrock2_lm, Mads.rosenbrock2_gradient_lm, [-1.2, 1.0], tolX=1e-12, tolG=1e-12)
res.minimum


using LsqFit
x0 = [100.0, 100.0]
@time res10 = LsqFit.lmfit(fopt, x0, Float64[]; autodiff=:forwarddiff, show_trace=true, maxIter=100)


res1.minimizer
res2.minimizer
res10.param

sum(fopt(res3.param).^2)


function f(x)

    F = similar(x)

    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)

    return F

end

function j(x)

    J = zeros(Number,2,2)

    J[1, 1] = x[2]^3-7
    J[1, 2] = 3*x[2]^2*(x[1]+3)
    u = exp(x[1])*cos(x[2]*exp(x[1])-1)
    J[2, 1] = x[2]*u
    J[2, 2] = u

    return J

end

x = [1., 2.]

f(x)
# 2-element Vector{Float64}:
#  22.0
#  -0.9622007558121382

j(x)

using ForwardDiff
ForwardDiff.jacobian(x -> f(x), x)

function jf(x)
  ForwardDiff.jacobian(x -> f(x), x)
end

x = [7., 9.]
f(x)
j(x)
jf(x)


function zjf(x)
    jacobian(f, x)[1]
end

jacobian(x -> f(x), x)[1]


function zjf(x)
    jacobian(f, x)[1]
end


zjf(x)


function jf(a)
    jacobian(a -> 100*a[1:3].^2, a)[1]
  end

# 2×2 Matrix{Number}:
#   1.0     48.0
#  -1.4806  -0.7403

Zygote.hessian(f, x)
Zygote.jacobian(f, x)



jacobian(a -> 100*a[1:3].^2, 1:7)[1] # first index (rows) is output
# 200    0    0  0  0  0  0
# 0  400    0  0  0  0  0
# 0    0  600  0  0  0  0

z = jacobian(a -> 100*a[1:3].^2, 1:7)
dim(z)
typeof(z)
length(z)
z[1]
z[2]
z[0]

function f(a)
    100*a[1:3].^2
end

a = 1:7
f(a)
jz = jacobian(a -> 100*a[1:3].^2, a)

function jf(a)
  jacobian(a -> 100*a[1:3].^2, a)[1]
end

function jf(a)
  jacobian(f, a)[1]
end

jf(a)
a = 1:7
jf(a)

a = 2:8
jf(a)

import Mads
import Test

import Distributed

callbacksucceeded = false
@Mads.stderrcapture function callback(x_best::AbstractVector, of::Number, lambda::Number)
	global callbacksucceeded
	callbacksucceeded = true
	println("The callback function was called: $x_best, $of, $lambda")
end

function callback(x_best::AbstractVector, of::Number, lambda::Number)
	global callbacksucceeded
	callbacksucceeded = true
	# println("The callback function was called: $x_best, $of, $lambda")
  println(of, " ", lambda)
end

Mads.madsinfo("Levenberg-Marquardt optimization of the Rosenbrock function with callback")
results = Mads.levenberg_marquardt(Mads.rosenbrock_lm, Mads.rosenbrock_gradient_lm, [0.0, 0.0]; show_trace=false, callbackiteration=callback)
results.minimum
@Test.test callbacksucceeded


using Optim, OptimTestProblems
UP = UnconstrainedProblems
prob = UP.examples["Extended Rosenbrock"]
Optim.optimize(UP.objective(prob), UP.gradient(prob), prob.initial_x, GradientDescent())

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


# function levenberg_marquardt(f::Function, g::Function, x0, o::Function=x->(x'*x)[1]; root::AbstractString="", tolX::Number=1e-4, tolG::Number=1e-6, tolOF::Number=1e-3, maxEval::Integer=1001, maxIter::Integer=100, maxJacobians::Integer=100, lambda::Number=eps(Float32), lambda_scale::Number=1e-3, lambda_mu::Number=10.0, lambda_nu::Number=2, np_lambda::Integer=10, show_trace::Bool=false, alwaysDoJacobian::Bool=false, callbackiteration::Function=(best_x::AbstractVector, of::Number, lambda::Number)->nothing, callbackjacobian::Function=(x::AbstractVector, J::AbstractMatrix)->nothing)
# 	# finds argmin sum(f(x).^2) using the Levenberg-Marquardt algorithm
# 	#          x
# 	# The function f should take an input vector of length n and return an output vector of length m
# 	# The function g is the Jacobian of f, and should be an m x n matrix
# 	# x0 is an initial guess for the solution
# 	# fargs is a tuple of additional arguments to pass to f
# 	# available options:
# 	#   tolX - search tolerance in x
# 	#   tolG - search tolerance in gradient
# 	#   tolOF - search tolerance in objective function
# 	#   maxIter - maximum number of iterations
# 	#   lambda - (inverse of) initial trust region radius
# 	#   lambda_mu - lambda decrease factor
# 	#   lambda_nu - lambda multiplication factor
# 	#   np_lambda - number of parallel lambdas to test
# 	#   show_trace - print a status summary on each iteration if true
# 	# returns: x, J
# 	#   x - least squares solution for x
# 	#   J - estimate of the Jacobian of f at x

## djb test
# %% djb test 6/15/2022

import Mads

function callback(x_best::AbstractVector, of::Number, lambda::Number)
  global callbacksucceeded
  callbacksucceeded = true
  println(of, " ", lambda)
end

ndim = 200

results = Mads.levenberg_marquardt(Mads.makerosenbrock(ndim), Mads.makerosenbrock_gradient(ndim), zeros(ndim),
  lambda_mu=2.0, np_lambda=10, show_trace=true, maxJacobians=1000, callbackiteration=callback)

results = Mads.levenberg_marquardt(Mads.makerosenbrock(ndim), Mads.makerosenbrock_gradient(ndim), results.minimizer, lambda_mu=0.1, np_lambda=10, show_trace=true, maxJacobians=10000, callbackiteration=callback, maxEval=1000000)
results.minimum

results = Mads.levenberg_marquardt(Mads.makerosenbrock(ndim), Mads.makerosenbrock_gradient(ndim), zeros(ndim),
  lambda_mu=2.0, np_lambda=10, show_trace=true, maxJacobians=1000)

