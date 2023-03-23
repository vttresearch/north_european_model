These Python scripts format and modify the timeseries data for district heating, hydropower, and electric demand.

Tested with Python 3.10.4.

## Usage

Rename the file Elec_demand/input/API_token_example.txt to API_token.txt and add your own Entsoe platform API-key to it (without spaces or extra lines). You can get the API-key by registering to the platform https://transparency.entsoe.eu/. 

Using Python, run **start_inputdata.py** which executes the python-files in src-directories under DH, Hydro, and Elec_demand folders and creates timeseries csv-files for those.

Run **copyIntermediateCSVs.py** to copy the created intermediate files to "timeseries/input" folder. This is needed to further process the timeseries into Backbone model format. Then proceed to the timeseries/ folder to do that.

Python requires installing of several packages to work correctly. These include:

- pandas
- matplotlib
- openpyxl
