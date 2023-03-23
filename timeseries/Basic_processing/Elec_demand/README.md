These python scripts format and modify the timeseries for electric demand from open power system data and data 
from ENTSO-E Transparency platform. Using two sources was needed as ENTSO-E Transparency platform does not include older data.

Tested with python 3.10.4.

Note: 

1. Values are not measured values for subregions between years (2011-2014): DK_1,DK_2,NO_1, NO_2, NO_3, NO_4, NO_5, SE_1, SE_2, SE_3, SE_4.
These values were calculated using the sums of those countries and the distribution generated from more recent measurements.

2. Values for CH were much lower during 2011-2014 than after 2015. 
They were adjusted so that the average from 2011-2014 is now the same as in 2015-2020.

## Usage

In order to create timeseries for electric demand run Elec_demand_timeseries.py. It executes the python files in src-folder.

Files in src-folder can be also executed separately in case intermediate results are needed. Elec_demand_timeseries.py deletes all the intermediate results after usage.

## Input

time_series_60min_singleindex.csv

File that contains hourly electricity demand (load) timeseries data from Data Platform – Open Power System Data (open-power-system-data.org) for selected countries and areas. Timeframe of downloaded data is 2011-2018 although the later years are incomplete.

API_token.txt

From https://transparency.entsoe.eu/ is downloaded programmatically years 2015 onwards.Add your own token into this file to access Entsoe transparency platform

constants.csv

Constants for modifying loads by constant or temperature. Modify according to needs.
C_update(t) = C_old(t) + C_0,alue + T_alue(t) * C_1,alue
where C_old(t) is the historical load, C_0,alue is a region dependent constant and C_1,alue is a region dependent temperature constant.

temperature.csv

Historical temperatues in Europe 

## Src

EntsoeQueryLoad.py

Fetches demand data from Entsoe transparency platform and stores them as area-specific files.
Smooths results and interpolates gaps.

EntsoeMergeLoadSummary.py

Merges individual areas specific files into one file containing years 2015-.

QueryLoad.py

Processes the input file and generates load data as area-specific files. 
Smooths results and interpolates gaps.

CreateAreaRates.py

Area values for DK, NO, and SE missing from year 2011-2014. The purpose of this code is to use the values from 2015-2020 and generate how sum of area is distributed to areas of these countries. 

FixLoadAreas.py

As input data for years 2011-2014 did not have areas for DK, NO, and SE, only sums, the distribution rates from years 2015-2020 were used to calculate the area values. 

MergeLoadSummary.py

Merges individual country files into one file 2011-2014. I

CombineLoadSummaries.py

Load summary files from 2011-2014 and 2015-2020 are combined to one file. As there is a jump in values for CH from one dataset to another, the earlier values are "corrected" to be more inline with later ones. Modifies loads depending on constants defined in constants.csv. 

## Output

summary_load_2011_2020_model_form_MWh.csv
