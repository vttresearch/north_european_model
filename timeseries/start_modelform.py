from src.LoadsToModelForm import conv_loads_tomodelform
from src.LimitsToModelForm import conv_limits_tomodelform
from src.LevelsToModelForm import conv_levels_tomodelform
from src.InflowToModelForm import conv_inflow_tomodelform
from src.GenerationToModelForm import conv_generation_tomodelform
from src.DHToModelForm import conv_dhdemand_tomodelform

# Convert intermediate timeseries files, created using the scripts in Basic_processing
# to the correct model format. 
# Output is printed to "timeseries/output"
# !! The software does not currently process EV time series from basic_processing/EV
# folder. Please copy them manually to Backbone input folder.

# select the regions for which you want timeseries values (others except DH)
selected_regions = ['FI00', 
                    'SE01', 'SE02', 'SE03', 'SE04', 
                    'NOS0', 'NOM1', 'NON1', 
                    'DE00', 
                    "DKE1", "DKW1",
                    "FR00",
                    "PL00",
                    "EE00",
                    "UK00",
                    "LT00",
                    "LV00",
                    "NL00",
                    "BE00",
                    "ES00"]


# call the conversion functions for different data 
#electrical load
conv_loads_tomodelform(selected_regions)

#hydro min and max limits
conv_limits_tomodelform(selected_regions)

#hydro historical reservoir levels
conv_levels_tomodelform(selected_regions)

#hydro inflow
conv_inflow_tomodelform(selected_regions)

#hydro min generation
conv_generation_tomodelform(selected_regions)

# select the regions for which you want DH timeseries values

selected_regions_dh = [
    'Helsinki',
    'Espoo',
    'Vantaa',
    'Turku',
    'Tampere',
    'Oulu',
    'Jyvaskyla',
    'FI00_others',
    'SE02', 'SE03', 'SE04',
    'DKE1',
    'DE00',
    'PL00']

# district heat demand time series to model format
conv_dhdemand_tomodelform(selected_regions_dh)
