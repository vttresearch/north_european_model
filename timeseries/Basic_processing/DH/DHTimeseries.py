"""
Processes DH timeseries calling functions in other py-files.
In the end, deletes all intermediate files.
Input:  DH summary.xlsx, Temperature.csv  
Output:  DH_2025_timeseries_summary.csv, DH_2030_timeseries_summary.csv
"""
from src.GenerateDHTimeseries import GenerateDH
from src.SummarizeDHs import DHSummary
from src.AverageDHs import DHAverage

 # 1 process timeseries data into separate area-specific files
DH = GenerateDH()
DH.process_areas()

# 2 combine areas to one file
summary = DHSummary()
summary.process_summary()

# 3 Remove intermediate results
summary.delete_intermediates()

# 4 Create 10%,50%,90% timerseries
average = DHAverage()
average.process_average()






