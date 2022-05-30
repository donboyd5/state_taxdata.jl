
module getTaxProblem

using CSV
using Tables
# using DataFrames
# using DelimitedFiles

function get_taxprob(stubnum)
    dir = "data/"
    stub = lpad(stubnum, 2, "0")

    fname = dir * "wh_" * stub * ".csv"
    wh = CSV.File(fname; header=false, types=Float64) |>
      Tables.matrix

    fname = dir * "geotargets_" * stub * ".csv"
    geotargets = CSV.File(fname; header=false, types=Float64) |>
      Tables.matrix

    fname = dir * "xmat_" * stub * ".csv"
    xmat = CSV.File(fname; header=false, types=Float64) |>
      Tables.matrix

    return (wh=wh, geotargets=geotargets, xmat=xmat)
end

end

# prob = get_taxprob(1)
# prob.wh
# prob.geotargets
# prob.xmat
