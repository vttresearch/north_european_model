These python scripts format and modify the timeseries for hydro minmax generation using ENTSO-E Hydropower modelling data (PECD). 

Tested with python 3.10.4.

## Usage

In order to create timeseries for hydro inflow run Reservoir_minmax_generation.py. It executes the python files in src-folder.

Files in src-folder can be also executed separately in case intermediate results are needed. Reservoir_minmax_generation.py deletes all the intermediate results after usage.

## Input

From hhttps://zenodo.org/record/3985078#.YgUKvurP2Ul:
PECD-hydro-weekly-reservoir-min-max-generation.csv

For Norway: PEMMDB_NOM1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NON1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NOS0_Hydro Inflow_SOR 20.xlsx

## Src

QueryMinMaxGeneration.py

Processes PECD-hydro-weekly-reservoir-min-max-generation.csv to generate individual hourly timeseries in areas-folder. Both downward and upward limits are given for each selected area. Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

QueryMinMaxGenerationNorway.py

Processes PEMMDB_NOM1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NON1_Hydro Inflow_SOR 20.xlsx, and PEMMDB_NOS0_Hydro Inflow_SOR 20.xlsx to generate hourly timeseries for Norway in areas-folder. Both downward and upward limits are given for each selected area. Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

SummarizeGeneration.py

Combines area-specific timeseries to summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv. Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

## Output

summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv
