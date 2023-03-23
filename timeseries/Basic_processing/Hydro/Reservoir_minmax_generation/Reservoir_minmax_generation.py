"""
Processes minmax generation calling functions in other py-files.
In the end, deletes all intermediate files.
Input: PECD-hydro-weekly-reservoir-min-max-generation.csv, PEMMDB_NOM1_Hydro Inflow_SOR 20.xlsx,
    PEMMDB_NON1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NOS0_Hydro Inflow_SOR 20.xlsx   
Output:  summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv
"""
from src.QueryMinMaxGeneration import QueryMinMaxGeneration
from src.QueryMinMaxGenerationNorway import QueryMinMaxGenerationNorway
from src.SummariseMinMaxGeneration import MinMaxSummary


# 1 process timeseries data into separate area-specific files
minmax = QueryMinMaxGeneration()
minmax.process_areas()

# 2 process timeseries data into separate area-specific files for Norway
minmaxnorway = QueryMinMaxGenerationNorway()
minmaxnorway.process_areas()

# 3 combine areas to one file
summary = MinMaxSummary()
summary.process_summary()

# 4 Remove intermediate results
summary.delete_intermediates()





