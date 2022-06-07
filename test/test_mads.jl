import Pkg
Pkg.rm("Mads")

Pkg.add("Mads")Mads.test()

import Pkg; Pkg.add(Pkg.PackageSpec(name="Mads", rev="master"))
Pkg.add("Gadfly")

import Mads
import Gadfly
Mads.test()
Mads.help()


Mads.functions()
# To list all the functions in a module, do:
# Mads.functions(BIGUQ)
Mads.functions("BIGUQ")
Mads.functions(Mads)

Mads.functions("get")
Mads.functions("leven")


@Mads.stderrcapture fopt(x) = [x[1], 2.0 - x[2]]
@Mads.stderrcapture gopt(x) = [1.0 0.0; 0.0 -1.0]

@time res1 = Mads.naive_levenberg_marquardt(fopt, gopt, [100.0, 100.0])
@Test.test LinearAlgebra.norm(results.minimizer - [0.0, 2.0]) < 0.01

@time res2 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], show_trace=true)
@Test.test LinearAlgebra.norm(results.minimizer - [0.0, 2.0]) < 0.01

@time res3 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], tolOF=1e-24)
@time res4 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], np_lambda=4, tolX=1e-8, tolG=1e-12)
# @time res5 = Mads.levenberg_marquardt(fopt, gopt, [100.0, 100.0], to

function callback(x_best::AbstractVector, of::Number, lambda::Number)
	global callbacksucceeded
	callbacksucceeded = true
	# println("The callback function was called: $x_best, $of, $lambda")
  println(of, " ", lambda)
end

ndim = 200
results = Mads.levenberg_marquardt(Mads.makerosenbrock(ndim), Mads.makerosenbrock_gradient(ndim), zeros(ndim),
  lambda_mu=2.0, np_lambda=10, show_trace=false, maxJacobians=1000, callbackiteration=callback)

results = Mads.levenberg_marquardt(Mads.makerosenbrock(ndim), Mads.makerosenbrock_gradient(ndim), results.minimizer,
  lambda_mu=2.0, np_lambda=10, show_trace=false, maxJacobians=1000, callbackiteration=callback)

println(results)

results.minimum
results.minimizer


results = Mads.levenberg_marquardt(Mads.makerosenbrock(ndim), Mads.makerosenbrock_gradient(ndim), zeros(ndim),
  tolOF=1e-99, tolX=1e-99, tolG=1e-99,
  lambda_mu=2.0, np_lambda=10, show_trace=false, maxJacobians=1000, callbackiteration=callback)

# ERROR: LoadError: MethodError: no method matching layer(::Vector{Gadfly.Geom.LineGeometry}, ::Vector{Gadfly.Geom.LineGeometry}, ::Vector{Gadfly.Geom.LineGeometry}, ::Vector{Gadfly.Geom.LineGeometry}, ::Vector{Gadfly.Geom.LineGeometry}; x=1:4, y=[0.9724953537852915, 0.955955284443903, 0.9008697445870358, 0.6321582462815516])
# Closest candidates are:
#   layer(::Any, ::Union{Function, Gadfly.Element, Gadfly.Theme, Type}...; mapping...) at C:\Users\donbo\.julia\packages\Gadfly\B5yQc\src\Gadfly.jl:169
# Stacktrace:
#  [1] plotseries(X::Matrix{Float64}, filename::String; nT::Int64, nS::Int64, format::String, xtitle::String, ytitle::String, title::String, logx::Bool, logy::Bool, keytitle::String, name::String, names::Vector{String}, combined::Bool, hsize::Measures.AbsoluteLength, vsize::Measures.AbsoluteLength, linewidth::Measures.AbsoluteLength, linestyle::Symbol, pointsize::Measures.AbsoluteLength, key_position::Symbol, major_label_font_size::Measures.AbsoluteLength, minor_label_font_size::Measures.AbsoluteLength, dpi::Int64, colors::Vector{String}, opacity::Float64, xmin::Nothing, xmax::Nothing, ymin::Nothing, ymax::Nothing, xaxis::UnitRange{Int64}, plotline::Bool, plotdots::Bool, firstred::Bool, lastred::Bool, nextgray::Bool, code::Bool,
#  returnplot::Bool, colorkey::Bool, background_color::Nothing, gm::Vector{Any}, gl::Vector{Any}, quiet::Bool, truth::Bool)
#    @ Mads C:\Users\donbo\.julia\packages\Mads\ZVE7t\src\MadsPlot.jl:1213
#  [2] top-level scope
#    @ C:\Users\donbo\.julia\packages\Mads\ZVE7t\test\miscellaneous.jl:174
# in expression starting at C:\Users\donbo\.julia\packages\Mads\ZVE7t\test\miscellaneous.jl:174
# in expression starting at C:\Users\donbo\.julia\packages\Mads\ZVE7t\test\runtests.jl:35