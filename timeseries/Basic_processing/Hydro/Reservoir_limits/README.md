These python scripts format and modify the timeseries for hydro minmax reservoir limits using ENTSO-E Hydropower modelling data (PECD). 

Tested with python 3.10.4.

## Usage

In order to create timeseries for hydro reservoir minmax limits run Reservoir_limits_timeseries.py. It executes the python files in src-folder.

Files in src-folder can be also executed separately in case intermediate results are needed. Reservoir_limits_timeseries.py deletes all the intermediate results after usage.

## Input

From https://zenodo.org/record/3985078#.YgUKvurP2Ul: PECD-hydro-weekly-reservoir-levels.csv and PECD-hydro-capacities.csv

For Norway: PEMMDB_NOM1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NON1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NOS0_Hydro Inflow_SOR 20.xlsx

## Src

QueryLimits.py

Processing reservoir limits and limits for ps_Open and ps_Closed. Only two years 2015 and 2016 included, 
as examples of non-leap year and leap-year. Inside the python-file Start and End values can be changed 
to modify the timeframe for produced timeseries.

QueryLimitsNorway.py

Processing reservoir limits for Norway as they were not in the original input file, psClosed included from capacity

SummariseLimits.py

Individual timeseries are combined into one file: summary_hydro_reservoir_limits_2015_2016_1h_MWh.csv. 
Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

## Output

summary_hydro_reservoir_limits_2015_2016_1h_MWh.csv
