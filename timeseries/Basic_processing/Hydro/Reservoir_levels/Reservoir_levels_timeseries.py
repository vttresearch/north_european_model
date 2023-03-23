"""
Processes historical hydro reservoir levels calling functions in other py-files.
In the end, deletes all intermediate files.
Input:  ts_reservoir_level.csv
Output file(s): ummary_historical_hydro_reservoir_levels_1h_MWh.csv
"""
from src.QueryReservoirLevels import QueryLevels
from src.SummariseLevels import LevelsSummary


# 1 process timeseries data into separate area-specific files
levels = QueryLevels()
levels.process_levels()

# 2 combine areas to one file
summary = LevelsSummary()
summary.process_summary()

# 3 Remove intermediate results
summary.delete_intermediates()





