These python scripts format and modify the distric heating timeseries.

Tested with python 3.10.4.

## Usage

In order to create timeseries for district heating consumption run DHTimeseries.py. It executes the python files in src-folder. Files in src-folder can be also executed separately in case intermediate results are needed. DHTimeseries.py deletes all the intermediate files after usage.

## Input

DHSummary.xlsx

Contains manually collected district heating data from different areas in Europe.The Production-worksheet contains values that are used as input in district heating calculations. The Production-worksheet summarises data from other worksheets. Each country has their separate worksheets that have been used to collect raw data from various sources. The initial data sources are identified in the country-specific worksheets. 

Note that there is some variation in the availability and reliability of source data. In fact, data of some of the countries were not used in the end at all as it was decided that DH is not relevant in those countries and/or there is not enough reliable source data available. 

In the Production-worksheet, for each country or area is defined the following parameters:

- DH: District heating production projections for years 2025 and 2030. Additionally values for 2010 to 2019 are adapted from 2025 projection.

- Space heating: Space heating values are simply result of subtracting water heating values from DH values.

- Water heating: Water heating is constant for all the years. It is estimated based on the average rate of water heating and projected DH production in 2025.


Temperature.csv

Contains temperature values for each country.

## Src

GenerateDHTimeseries.py

Transforms DHSummary.xlsx and Temperature.xlsx into area-specific csv-files. Produced csv-files contain 
two hourly timeseries where yearly production values are distributed based on temperatures at each hour. There is timeseries both for 2025 and 2030 values. Variation in temperatures is used only for the Space heating part and the amount of water heating is considered as constant. Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

SummarizeDHs.py

Combines area-specific timeseries into two year-specific csv-files: DH_2025_timeseries_summary.csv and 
DH_2030_timeseries_summary.csv. Inside the python-file Start and End values can be changed to modify the timeframe for produced timeseries.

## Output

DH_2025_timeseries_summary.csv and DH_2030_timeseries_summary.csv
