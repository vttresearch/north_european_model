"""
Processes all the timeseries under Basic_processing
Note: rename the file Elec_demand/input/API_token_example.txt to API_token.txt and add your own entsoe platform API-key to it
Otherwise you cannot use EntsoeQueryLoad.
"""
from DH.src.GenerateDHTimeseries import GenerateDH
from DH.src.SummarizeDHs import DHSummary
from DH.src.AverageDHs import DHAverage
from Hydro.Inflow.src.QueryInflow import QueryInflow
from Hydro.Inflow.src.SummarizeInflows import InflowSummary
from Hydro.Inflow.src.AverageInflows import AverageInflow
from Hydro.Reservoir_limits.src.QueryLimits import QueryMinMaxLimits
from Hydro.Reservoir_limits.src.QueryLimitsNorway import QueryMinMaxLimitsNorway
from Hydro.Reservoir_limits.src.SummariseLimits import LimitsSummary
from Hydro.Reservoir_minmax_generation.src.QueryMinMaxGeneration import QueryMinMaxGeneration
from Hydro.Reservoir_minmax_generation.src.QueryMinMaxGenerationNorway import QueryMinMaxGenerationNorway
from Hydro.Reservoir_minmax_generation.src.SummariseMinMaxGeneration import MinMaxSummary
from Hydro.Reservoir_levels.src.QueryReservoirLevels import QueryLevels
from Hydro.Reservoir_levels.src.SummariseLevels import LevelsSummary
from Elec_demand.src.EntsoeQueryLoad import EntsoeQueryLoad
from Elec_demand.src.EntsoeMergeLoadSummary import EntsoeMergeAreaLoads
from Elec_demand.src.QueryLoad import QueryLoad
from Elec_demand.src.CreateAreaRates import AreaRates
from Elec_demand.src.FixLoadAreas import FixLoadAreas
from Elec_demand.src.MergeLoadSummary import MergeAreaLoads
from Elec_demand.src.CombineLoadSummaries import CombineSummaries
from Elec_demand.src.AverageLoads import AverageLoads
import os
import glob

ADD="DH/"
# process timeseries data into separate area-specific files
DH = GenerateDH(ADD)
DH.process_areas()

# combine areas to one file
DHsummary = DHSummary(ADD)
DHsummary.process_summary()

# remove intermediate results
DHsummary.delete_intermediates()

# create 10%,50%,90% timerseries
average = DHAverage(ADD)
average.process_average()

ADD="Hydro/Inflow/"
# process timeseries data into separate area-specific files
inflow = QueryInflow(ADD)
inflow.process_inflow()

# combine areas to one file
inflowsummary = InflowSummary(ADD)
inflowsummary.process_summary()

# calculate "average" year
average = AverageInflow(ADD)
average.process_average()

# remove intermediate results
inflowsummary.delete_intermediates()

ADD="Hydro/Reservoir_limits/"
# process timeseries data into separate area-specific files
limits = QueryMinMaxLimits(ADD)
limits.process_areas()

# process timeseries data into separate area-specific files for Norway
limitsNorway = QueryMinMaxLimitsNorway(ADD)
limitsNorway.process_areas()

# combine areas to one file
limitsummary = LimitsSummary(ADD)
limitsummary.process_summary()

# remove intermediate results
limitsummary.delete_intermediates()

ADD="Hydro/Reservoir_levels/"
# process timeseries data into separate area-specific files
levels = QueryLevels(ADD)
levels.process_levels()

# combine areas to one file
levelssummary = LevelsSummary(ADD)
levelssummary.process_summary()

# remove intermediate results
levelssummary.delete_intermediates()

ADD="Hydro/Reservoir_minmax_generation/"
# process timeseries data into separate area-specific files
minmax = QueryMinMaxGeneration(ADD)
minmax.process_areas()

# process timeseries data into separate area-specific files for Norway
minmaxnorway = QueryMinMaxGenerationNorway(ADD)
minmaxnorway.process_areas()

# combine areas to one file
gensummary = MinMaxSummary(ADD)
gensummary.process_summary()

# remove intermediate results
gensummary.delete_intermediates()

ADD="Elec_demand/"
files = glob.glob(ADD+'output/*')
for f in files:
    os.remove(f)

# process timeseries data into separate area-specific files (ENTSO-E Transparency platform) 
entsoeloads = EntsoeQueryLoad(ADD)
entsoeloads.process_areas()

# summarise areas from ENTSO-E Transparency platform
entsoemerged = EntsoeMergeAreaLoads(ADD)
entsoemerged.process_areas()
entsoemerged.delete_intermediates()

# process timeseries data into separate area-specific files (Open power system data) 
loads = QueryLoad(ADD)
loads.process_areas()

# process rates based on years 2015-2020 for getting distribution of loads for areas in SE, NO, DK in 2011-2014
#Note: measured values were not available for subregions: DK_1,DK_2,NO_1,NO_2, NO_3, NO_4, NO_5, SE_1, SE_2, SE_3, SE_4.
#These values are calculated using the sums of those countries and the distribution generated from more recent measurements.
rates = AreaRates(ADD)
rates.NonLeapAreaRates()
rates.LeapAreaRates()

# fix distribution of loads for subregions in SE, NO,and DK in 2011-2014 
fix = FixLoadAreas(ADD)
fix.process_fix()

# summarise areas for years 2011-2014 (Open power system data)
merged = MergeAreaLoads(ADD)
merged.process_areas()

# combine summaries from 2011-2014 and 2015-2020
# Note: big differnece in early and later values for CH, consequently the older ones were
# "corrected" to be more inline with later ones
# Scenario can be also added to the call to update historical loads.
summary = CombineSummaries(ADD)        
summary.process_summary()

# remove intermediate results
merged.delete_intermediates()
fix.delete_intermediates()
summary.delete_intermediates()

# create 10%,50%,90% timeseries for one year
averages = AverageLoads(ADD)        
averages.process_average()

