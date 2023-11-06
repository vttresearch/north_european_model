"""
Processes elec demands calling functions in other py-files.
In the end, deletes all intermediate files.
Input: time_series_60min_singleindex.csv (Open power system data), data from ENTSO-E Transparency platform
Output file(s): summary_2011-2020-1h.csv 
"""
from src.EntsoeQueryLoad import EntsoeQueryLoad
from src.EntsoeMergeLoadSummary import EntsoeMergeAreaLoads
from src.QueryLoad import QueryLoad
from src.CreateAreaRates import AreaRates
from src.FixLoadAreas import FixLoadAreas
from src.MergeLoadSummary import MergeAreaLoads
from src.CombineLoadSummaries import CombineSummaries
from src.AverageLoads import AverageLoads
import os
import glob

#choose scenario and year
scenario = "Distributed Energy"
year = 2040
#scenario = "National Trends"
#year = 2025

#option to redownload ENTSO-E Transparency platform datafiles
options =  {"download": False}

# Clean output folder
files = glob.glob('output/*')
for f in files:
    os.remove(f)

# 1 Download electrical load timeseries data into separate area-specific files (ENTSO-E Transparency platform) 
# options["download"] governs if downloads are done or if old results are used
# Note: create the file input/API_token.txt and add your own entsoe platform API-key to input/API_token.txt
entsoeloads = EntsoeQueryLoad()
if options["download"] == True:
    entsoeloads.process_areas()

# 2 summarise areas from ENTSO-E Transparency platform
entsoemerged = EntsoeMergeAreaLoads()
entsoemerged.process_areas()
entsoemerged.delete_intermediates()

# 3 process timeseries data into separate area-specific files (Open power system data) 
loads = QueryLoad()
loads.process_areas()

# 4 process rates based on years 2015-2020 for getting distribution of loads for areas in SE, NO, DK in 2011-2014
#Note: measured values were not available for subregions: DK_1,DK_2,NO_1,NO_2, NO_3, NO_4, NO_5, SE_1, SE_2, SE_3, SE_4.
#These values are calculated using the sums of those countries and the distribution generated from more recent measurements.
rates = AreaRates()
rates.NonLeapAreaRates()
rates.LeapAreaRates()

# 5 fix distribution of loads for subregions in SE, NO,and DK in 2011-2014 
fix = FixLoadAreas()
fix.process_fix()

# 6 summarise areas for years 2011-2014 (Open power system data)
merged = MergeAreaLoads()
merged.process_areas()

# 7 combine summaries from 2011-2014 and 2015-2020
# Note: big differnece in early and later values for CH, consequently the older ones were
# "corrected" to be more inline with later ones
#summary = CombineSummaries()        
# This function also takes into account scenario and year to modify loads.
# Run without arguments to pass modifications.
summary = CombineSummaries("",scenario, year) 
summary.process_summary()

# 8 Remove intermediate results
merged.delete_intermediates()
fix.delete_intermediates()
summary.delete_intermediates()

# 9 Create 10%,50%,90% timeseries for one year
averages = AverageLoads()        
averages.process_average()





