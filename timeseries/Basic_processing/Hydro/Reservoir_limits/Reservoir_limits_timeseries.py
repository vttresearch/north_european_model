"""
Processes hydro minmax levels calling functions in other py-files.
In the end, deletes all intermediate files.
Input:PPECD-hydro-weekly-reservoir-levels.csv, PECD-hydro-capacities.csv,EMMDB_NOM1_Hydro Inflow_SOR 20.xlsx,
        PEMMDB_NON1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NOS0_Hydro Inflow_SOR 20.xlsx  
Output file(s): summary_hydro_reservoir_levels_2015_2016_1h_MWh.csv 
"""
from src.QueryLimits import QueryMinMaxLimits
from src.QueryLimitsNorway import QueryMinMaxLimitsNorway
from src.SummariseLimits import LimitsSummary


# 1 process timeseries data into separate area-specific files
limits = QueryMinMaxLimits()
limits.process_areas()

# 2 process timeseries data into separate area-specific files for Norway
limitsNorway = QueryMinMaxLimitsNorway()
limitsNorway.process_areas()

# 3 combine areas to one file
summary = LimitsSummary()
summary.process_summary()

# 4 Remove intermediate results
summary.delete_intermediates()





