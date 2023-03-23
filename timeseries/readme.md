These Python scripts format and modify the ENTSO-E time series for electrical load, hydropower, and district heating. Each folder contains a separate readme explaining the data processing and requirements in greater detail.

District heating data is for years 2010-2019, 2025 and 2030 projections. Hydropower data is for years 1982-2016 (some countries also 2017), historical reservoir levels data is extrapolated from years 2015-2022 May. Electrical load data is for years 2011-2019.

Tested with Python 3.10.4.

## Usage

* run **start_inputdata.py** in basic_processing folder to create the uniform timeseries-files from the original input data. See instructions in basic_processing/readme.md.
* The results from previous step are stored in individual output-folders where they should be transferred manually to "timeseries/input" 
* run **start_modelform.py** to convert intermediate timeseries files to the correct model format. Output is printed to "timeseries/output"
