# Europe Input

Input data conversions for the North European energy system model for the Backbone model.


## Preparation

Make sure you have [Backbone](https://gitlab.vtt.fi/backbone/backbone)  installed. For the moment we recommend using the master branch.

The usual preparations for the Backbone installation include:

* copy Backbone/1_options_temp.gms to Backbone/input and rename it 1_options.gms


## Installation

Clone the repository on your machine. Make sure you have [Julia](https://julialang.org/)  version >= 1.5 installed. Install the dependencies by starting Julia interactive session (e.g. typing Julia in command prompt) in "europe-input" folder and running following commands:

	using Pkg
	Pkg.activate(".") 
    Pkg.instantiate()


Note: Building PyCall might be slow, but do not interrupt it. 

The timeseries/ folder contains Python scripts for which you need a Python installation. More instructions to be added.


## Getting the input data

The conversion does not work in isolation, thus you first need the proper input data. The Following VRE time series ([10.5281/zenodo](https://doi.org/10.5281/zenodo.3702418)) should be added in the input/vre/ folder manually as files are really large: 

* PECD-MAF2019-wide-PV.csv
* PECD-MAF2019-wide-WindOffshore.csv
* PECD-MAF2019-wide-WindOnshore.csv

Note that these are older results with very low capacity factors. There is an option to use newer results but the processing of the input files is not included in this repository.

Electrical load data is usually downloaded from the ENTSO-E transparency platform. User-specific API token is needed for automatically accessing the more recent data in ENTSO-E Transparency platform. The token should be added to API_token.txt-file. Alternatively, if you can obtain the files from another source, you can skip downloads by setting the relevant option in **start_inputdata.py**.


## Usage

The package contains functions for building both the main excel input file for Backbone containing the energy system description, functions for building the necessary time series, and example GAMS run files.

### Building the main input file
For building the Excel input file for Backbone, run the script **starthere.jl**. starthere.jl contains a detailed explanation of the function argument along with examples. By default the output file will be **bb_input1-3x.xlsx**.

### Building the time series files

Time series are built in two steps: for VRE and others (electricity, hydro, and district heat). 

For VRE, the series are converted into Backbone format by running the function **convert_vre**, the call is found in the file **starthere.jl**. Note: PyCall might use different python version, that does not find all packages. Current workround is to run using PyCall; pyimport_conda("pandas", "pandas")


For electricity, hydro, and district heating, Python language is needed.  Python requires installing of several packages to work correctly. Recommend approach is to set up an environment in Anaconda (https://www.anaconda.com/products/distribution). You can set up the environment in **timeseries/** folder by running the following command line commands

    conda env create -f environment.yml
    conda activate hopetimeseries
    
    
Then:

1. run **start_inputdata.py** (folder timeseries/basic_processing/) to create the uniform timeseries-files from the original input data. In order to be able to do this, you should acquire your personal API-key to entsoe platform. API-key should be put into file "timeseries/basic_processing/elec_demand/input/API_token.txt". 
2. run **copyIntermediateCSVs.py** to copy created intermediate files to "timeseries/input" folder
3. run **start_modelform.py** (folder timeseries) to convert intermediate timeseries files to the correct model format. Output is printed to "timeseries/output" folder

Each subfolder in *basic_processing*  contains a separate readme file explaining the data processing and requirements in greater detail. Tested with python 3.10.4.

### Giving the inputs to Backbone

Move the main Excel input file (e.g. bb_input1-3x.xlsx) to Backbone input folder. Also move all .csv files from  **output/** and **timeseries/output** folders to Backbone input folder.

### Manual additions

While we are working on the input data to be comprehensive, there are still things you need to do manually. It is suggested that you copy the rows from file **manual_additions/additional_backbone_input** to the respective tables in the backbone input file bb_input1.xlsx. Specific suggested manual additions also include

* Set fixed start levels to water reservoirs, which are listed in **manual_additions/additional_backbone_input** (sheet Boundary properties for states). You would do this by putting "1" in the boundStart column and rows pertaining to the water reservoirs of p_gn sheet in the backbone input file bb_input1.xlsx.

* Set the operation mode for conversion units are modified in changes.inc. In the current input data, all units are set as **directoff**, but changes.inc converts certain units to LP or MIP.

### Setting up run files before running Backbone

Working model requires certain run specificaion files for GAMS

* Copy **changes.inc** and **timeandsamples.inc** from GAMS_files folder to Backbone input folder. Note: changes.inc calls csv2gdx and some older versions of csv2gdx do not support very long time series. In this case, install also a more recent gams and manually add a hard coded file path to changes inc, e.g. converting `$call 'csv2gdx ...` to `$call 'c:\GAMS\38\csv2gdx ...`
* Use a suitable modelsInit.gms and scheduleinit.gms file, for example copy **modelsInit_example.gms** and **scheduleInit_example.gms** from GAMS_files to Backbone input folder and rename them to modelsInit.gms and scheduleInit.gms. 
* Optional: copy **cplex.opt** from GAMS_files folder to backbone root folder. 
* Additional input data can be given by creating **input/bb_input_addData1.xlsx** file. A mandatory structure is that the file has all tables of a full input file except ts_cf, ts_inlux, and ts_node. This option enables adding scenarios, modifying input data without the need to alter the automatically generated file, and saving your own modifications while generating a new base input data file.

### Running the Backbone model

Run the model by running Backbone.gms in GAMS. The model requires following command line options (use two hyphens in the beginning of each)

* **--input_file_excel** is a mandatory parameter defining the used input excel file name (e.g. bb_input1-3x.xlsx)
* **--modelledDays** is a mandatory parameter defining how many days the model runs. Currently capped between [1-365] if used in combination with tsYear. Otherwise user can give longer time periods, but must check that original timeseries length will not be exceeded.
* **--tsYear** is a mandatory parameter with allowed values of 0 or [xxxx - 2019]. Selecting a specific year greatly reduce the model size, but the model will use time series only from the selected year and loop those time series. The Model does always model e.g. year 2025, but time series year changes time series profiles and annual demands and water inflows.  

Working command line options for backbone.gms would, for an example, be

	--input_file_excel=bb_input1-3x.xlsx --tsYear=0
	--input_file_excel=bb_input1-3x.xlsx --modelYear=2030 --tsYear=2011 --forecasts=2
	--input_file_excel=bb_input1-3x.xlsx --modelledDays=30 --tsYear=2015 --priceMultiplier=0.8 

J

The model supports few user given additional options. All these are optional. Behaviour and default values are listed below.
* --modelYear [2025, 2030]. Default 2025. Allows a quick selection of the the modelled year. Currently two options and impacts only district heating demand. 
* --tsYear  allows a quick selection of which time series year the model uses. Giving this parameter greatly reduces the solve time as the model drops ts data from other years. Default value = none, which means that the model will use full time series and user is responsible for giving the correct starting time step.
* --forecasts [none, 2, 4]. Default value: none. **Recommended value: 2**. Enables reading forecast timeseries and activating relevant sections in the model. Currently Accepted values are none (realized values only), 2 (realized values and 1 central forecast), or 4 (realized values, 1 central forecast, 1 difficult forecast, 1 easy forecast). In difficult, we use low hydro inflow, high demand inflows, and average VRE. 
* --modelledDays [1-365] defines the amount of days the model runs. If not given, the default value is 365. If used with tsYear, the maximum value is 365. 
* --priceMultiplier [positive float] allows a quick command line adjustmenet of fuel and emission prices. The multiplier is applied fully to fuel prices and half to emission prices. E.g priceMultiplier = 0.7 means that fuel price are -30% and emission price -15%.



## Support

Contact the authors.

## Authors and acknowledgment
Jussi Ikäheimo - model development, time series, testing
Anu Purhonen - time series
Tomi J. Lindroos - time series, testing

## License
To be specified.

