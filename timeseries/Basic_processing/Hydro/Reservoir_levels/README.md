These python scripts format and modify the timeseries for historical hydro reservoir levels using ENTSO-E transparency platform data. 

Tested with python 3.10.4.

## Usage

In order to create timeseries for hydro reservoir levels run Reservoir_levels_timeseries.py. It executes the python files in src-folder.

Files in src-folder can be also executed separately in case intermediate results are needed. Reservoir_levels_timeseries.py deletes all the intermediate results after usage.

## Input

From entsoe transparency platform:
ts_reservoir_level.csv


## Src

QueryLevels.py

Processes horly timeseries for individual areas in input file. Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

SummariseLevels.py

Individual timeseries are combined into one file: summary_historical_hydro_reservoir_levels_1h_MWh.csv. 
Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

## Output

summary_historical_hydro_reservoir_levels_1h_MWh.csv
