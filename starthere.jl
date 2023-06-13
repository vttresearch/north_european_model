# The file contains the specification of settings and function calls 
# for the plant portfolio
# creation and VRE capacity factor creation for Backbone model. 

using Revise
using EuropeInput


# specify the input file names 
filenames = Dict()
options = Dict()

# input files for unit capacities
# uncomment the ones needed
filenames["plantsourcefiles"] = ["input/capacity/TYNDP-2020-capacities.xlsx", 
                                "input/capacity/additional-units-chp.xlsx", #CHP
                                #"input/capacity/additional-units-vre.xlsx", #updates to VRE capacity
                                #"input/capacity/additional-units-conventional.xlsx" #updates to some conventional units
                                ]


# input file for transmission line capacities
filenames["linesourcefiles"] = ["input/TYNDP-2020-Scenario-Datafile.xlsx", "input/additional-lines1.xlsx"]
# input file for unit technical data
filenames["techdatafile"] = "input/maf2020techdata.xlsx"
# input file of unit types and their model names
filenames["unitdictfile"] = "input/unitdict.xlsx"
# input file of constant commodity demands
filenames["demandfile"] = "input/demand.xlsx"
# input file of commodity prices
filenames["fuelpricefile"] = "input/fuelprices.xlsx"

# Specify the output files
# Main backbone input template
# bb_input1.xlsx for Backbone version 2.x
# bb_input1-3x-example.xlsx for Backbone master branch
#filenames["bb_base_file"] = "output/bb_input1-2x-example.xlsx"
filenames["bb_base_file"] = "output/bb_input1-3x-example.xlsx"

# Completed main backbone input file
filenames["bbinputfile"] = "output/bb_input1-3x.xlsx"

# capacity factor times series filename
filenames["bbtimeseriesfile"] = "output/bb_ts_cf_io.csv"

# input folder of VRE time series (in entso-e format)
filenames["timeseriesfolder"] = "input/vre/"

# define backbone version (2.x or 3.x)
# make sure that the input file template (filenames["bbinputfile"]) is compatible with the specified version!
options["bb_version"] = "3.x"

# set true if you wish to use PECD 2021 data for selected VRE series 
options["use_updated_VRE_series"] == false

# set true if parameters for special plant start types are included
options["use_constrained_starttypes"] = false

# define simulation scenario and year
scenario = "National Trends" 
year = 2025


# Define nodes for which 
# - plants are included 
# - between which lines are included
# - VRE capacity factor time series are included 
includenodes = ["FI00", 
                "SE01", "SE02", "SE03", "SE04", 
                "NOS0", "NOM1", "NON1", 
                "DE00", 
                "DKE1", "DKW1",
                "FR00",
                "PL00",
                "EE00",
                "UK00",
                "LV00",
                "LT00",
                "NL00"]

# convert entso-e unit and line data into Backbone excel
convert_entsoe(includenodes, filenames, options, scenario, year)

# convert VRE time series into CSV "ts_cf_io"
convert_vre(includenodes, filenames, options)

# produce a dummy heat demand for FI00_others_dheat region and industrial process heat
# also produce hydrogen demand for certain nodes
make_other_demands(filenames, scenario, year)

