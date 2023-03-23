"""
Processes hydro inflows calling functions in other py-files.
In the end, deletes all intermediate files.
Input: PECD-hydro-daily-ror-generation.csv,PECD-hydro-weekly-inflows.csv 
Output file(s): summary_hydro_inflow_1982-2020_1h_MWh.csv,
    summary_hydro_average_year_1982-2020_1h_MWh.csv 
"""
from src.QueryInflow import QueryInflow
from src.SummarizeInflows import InflowSummary
from src.AverageInflows import AverageInflow


# 1 process timeseries data into separate area-specific files
inflow = QueryInflow()
inflow.process_inflow()

# 2 combine areas to one file
summary = InflowSummary()
summary.process_summary()

# 3 execute average years
average = AverageInflow()
average.process_average()

# 4 Remove intermediate results
summary.delete_intermediates()





