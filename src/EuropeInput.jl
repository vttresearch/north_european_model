module EuropeInput

using XLSX
using DataFrames
using Printf
using Dates
using Statistics

export convert_entsoe, convert_vre, make_other_demands

include("utilities.jl")
include("timeseries.jl")
include("maketables.jl")
include("readinput.jl")
include("disaggregate.jl")
include("convertentso.jl")

# global variables
#time origin
timeorigin = DateTime(1982, 1, 1)

#electricity node suffix
elec_suffix = "_elec"

end # module
