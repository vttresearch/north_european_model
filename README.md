# North European energy system model

This repository contains North European energy system model. The model is built for the Backbone modelling framework. 

This readme has following main sections
- Installing Backbone and the North European Model
- Installing Julia, Conda, and setting up the environments
- Downloading required time series files
- Building input files for backbone
- Running Backbone



## Installing Backbone and the North European Model

Make sure you have Backbone installed. For the moment, the North European Model works only in the Backbone master branch. 

If you do not have Backbone, use git to clone the Backbone from [release-3.x](https://gitlab.vtt.fi/backbone/backbone/). Master is the default branch, but you can use git to [switch between braches](https://gitlab.vtt.fi/backbone/backbone/-/wikis/Versions).

Use git to clone the repository of North European model. The easiest approach to install North European model under Backbone, e.g. c:\backbone\north_european_model.


## Installing Julia, Conda, and setting up the environments

Make sure you have [Julia](https://julialang.org/)  version >= 1.5 installed. Install the Julia dependencies by starting Julia interactive session (e.g. typing 'Julia' in command prompt) in **north_european_model/** folder and running following commands:

	using Pkg
	Pkg.activate(".") 
    Pkg.instantiate()

Note: Building PyCall might be slow, but do not interrupt it. 

The recommend approach to install Python dependencies is to set up a new environment in [Anaconda](https://www.anaconda.com/products/distribution). New users might want to use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) instead of more complicated Anaconda. 

Open the installed conda, go to folder **timeseries/**, and set up the environment by running the following commands

    conda env create -f environment.yml
    conda activate northEuropeanModel

Installed environment does not contain [entsoe-py](https://github.com/EnergieID/entsoe-py) as conda does not automatically find it. After creating and activating the northEuropeanModel environment, install entsoe-py by following command:

    python -m pip install entsoe-py
   
ENTSO-E user-specific API token is needed for automated data queries from the ENTSO-E Transparency platform. To set up your API key
* go to ENTSO-E transparency platform, https://transparency.entsoe.eu/dashboard/show
* create a user account by clicking login at the top right of the page
* request API by sending and email to transparency@entsoe.eu with “Restful API access” in the subject line
* create file timeseries\Basic_processing\Elec_demand\input\API_token.txt and add the received token there. 

After these steps, you should have required softwares and environments ready. 

Python scripts are tested with python 3.10. Python environment is tested with miniconda.


## Downloading required time series files

The North Europe model has extensive time series that are too large to be shared in this repository. Time series data are collected in two steps: for VRE (wind, solar) and others (electricity, hydro, and district heat). 

**VRE time series** are from ENTSO-E PECD dataset ([10.5281/zenodo](https://doi.org/10.5281/zenodo.3702418)). Following files are large and should be added in the **input/vre/** folder manually: 
* PECD-MAF2019-wide-PV.csv
* PECD-MAF2019-wide-WindOffshore.csv
* PECD-MAF2019-wide-WindOnshore.csv

Note: that these are older results with very low capacity factors. There is an option to use newer results but the processing of the input files is not yet included in this repository.

**Other time series data** are shared in this repository or queried from the ENTSO-E transparency platform. 

* Open Conda and activate the northEuropeModel environment
* go to folder **timeseries\Basic_processing**
* run start_inputdata.py (type 'python start_inputdata.py') to create the uniform timeseries-files from the original input data. The run time is from 20 to 40 min and requires internet connection.
* run copyIntermediateCSVs.py to copy created intermediate files to **timeseries/input** folder. These are not model input, but intermediate storages for downloaded data.

If needed, each subfolder in timeseries\basic_processing\ contains a separate readme file explaining the data processing and requirements in greater detail.

Note: If you receive errors from missing packages, API key, or other, see previous section "Installing Julia, Conda, and setting up the environments".

Note: Basic processing has EV folder, but that is still work in progress and cannot be run with the currently shared files.


## Building and copying input files for backbone

The package contains functions for building 
* the main excel input file for Backbone containing the energy system description (Julia), 
* VRE time series csv files (Julia),
* other time series csv files (Python)

Time series are built in two steps: for VRE (wind, solar) and others (electricity, hydro, and district heat). 

### Running julia for bb-input and VRE timeseries

The file 'starthere.jl' in the north_european_model root folder contains three main functions 'convert_entsoe', 'convert_vre', and 'make_other_demands'. Check that all of these are active (not commented out).

run the script **starthere.jl**, e.g. typing 'Julia' in command prompt in "europe-input" folder and running following commands:

	using Pkg
	Pkg.activate(".") 
    Pkg.instantiate()
    include("starthere.jl")

In certain cases, Julia's PyCall might use different python version, that does not find all packages. If julia crashes complaining, e.g. that pandas is not found, run 

	using PyCall
	pyimport_conda("pandas", "pandas")

And the then rerun starthere.jl.

By default, the output files will be 
* ./output/bb_input1-3x.xlsx
* ./output/bb_ts_cf_io.csv
* ./output/bb_ts_cf_io_10p.csv
* ./output/bb_ts_cf_io_50p.csv
* ./output/bb_ts_cf_io_90p.csv

Copy these files to **backbone/input** folder


### Running python for other timeseries

Open Conda and activate the northEuropeModel environment. Then

* go to folder **./timeseries**
* run start_modelform.py to convert intermediate timeseries files to the correct model format. 
* Output csv files are printed to **./timeseries/output** folder. Copy these files to **backbone/input**

At the time of writing, the total size of files from Julia and Python are slightly below 2 Gb. 


### Manual additions

While we are working on the input data to be comprehensive, there are still some data that you need to do add manually to the input excel. 
* open **additional backbone input.xlsx** in .\north_european_model\manual_additions\
* copy the data to the **bb_input1-3x.xlsx** in backbone/input folder

Current manual additions increase FR demand response capacity, set upwardLimits for ROR storages and converts them to constant upwardLimit, changes allow spillages in hydro nodes, and set certain reference values and balance penalties. Each change is accompanied with additional info 'replace' or 'add' indicating whether user should edit an existing value or add a new row to data tables.



### Run specification files


Working model requires certain run specificaion files for GAMS

* copy **cplex.opt** from GAMS_files folder to backbone root folder. 
* Use a suitable **modelsInit.gms** and **scheduleInit.gms** file, for example copy **modelsInit_example.gms** and **scheduleInit_example.gms** from GAMS_files to backbone\input folder and rename them to modelsInit.gms and scheduleInit.gms. 
* Copy **1_options.gms**, **changes.inc** and **timeandsamples.inc** from GAMS_files folder to Backbone input folder. 

Note: included scheduleInit file has a specific structure that it works with *tsYear* and *modelledDays* parameters. If using your own file, adapt a similar structure to the file.

Note: changes.inc does quite many different tasks to read and process input data, see the content and introduction at the beginning of the file. The file has a separate section where user can do additional changes.

Note: changes.inc calls csv2gdx and some older versions of csv2gdx do not support very long time series. In this case, install also a more recent gams and manually add a hard coded file path to changes inc, e.g. converting `$call 'csv2gdx ...` to `$call 'c:\GAMS\45\csv2gdx ...`


Additional input data can be given by creating **input/bb_input_addData1.xlsx** file. A mandatory structure is that the file has all tables of a full input file except ts_cf, ts_inlux, and ts_node. This option enables adding scenarios, modifying input data without the need to alter the automatically generated file, and saving your own modifications while generating a new base input data file.


## Running Backbone

Run the model by running Backbone.gms in GAMS. The model requires following command line options (use two hyphens in the beginning of each)

* **--input_file_excel** is a mandatory parameter defining the used input excel file name (e.g. bb_input1-3x.xlsx)
* **--modelledDays** is a mandatory parameter defining how many days the model runs. Currently capped between [1-365] if used in combination with tsYear. Otherwise user can give longer time periods, but must check that original timeseries length will not be exceeded.
* **--tsYear** is a mandatory parameter with allowed values of 0 or [xxxx - 2019]. Selecting a specific year greatly reduce the model size, but the model will use time series only from the selected year and loop those time series. The Model does always model e.g. year 2025, but time series year changes time series profiles and annual demands and water inflows.  

Working command line options for backbone.gms would, for an example, be

	--input_file_excel=bb_input1-3x.xlsx --tsYear=0
	--input_file_excel=bb_input1-3x.xlsx --modelYear=2030 --tsYear=2011 --forecasts=2
	--input_file_excel=bb_input1-3x.xlsx --modelledDays=30 --tsYear=2015 --priceMultiplier=0.8 

The model supports few user given additional options. All these are optional. Behaviour and default values are listed below.
* --modelYear [2025, 2030]. Default 2025. Allows a quick selection of the the modelled year. Currently two options and impacts only district heating demand. 
* --tsYear [0, 2011-2016]. Default 2015. allows a quick selection of which time series year the model uses. Giving this parameter greatly reduces the solve time as the model drops ts data from other years. By giving value 0, user can run the model with multiyear time series, but the user is responsible for giving the correct starting time step and checking for error. This feature (tsYear=0) is untested.
* --modelledDays [1-365]. Default 365. This option defines the amount of modelled days. If used with tsYear, the maximum value is 365. 
* --forecasts [1, 2, 4]. Default value: 4. Activates forecasts in the model and requires 10p, 50p, and 90p time series filen in the input folder. Currently Accepted values are 1 (realized values only), 2 (realized values and 1 central forecast), or 4 (realized values, 1 central forecast, 1 difficult forecast, 1 easy forecast). It is recommended to use 4 forecasts due to improved hydro power modelling. 
* --priceMultiplier [positive float]. Default 1. allows a quick command line adjustmenet of fuel and emission prices. The multiplier is applied fully to fuel prices and half to emission prices. E.g priceMultiplier = 0.7 means that fuel price are -30% and emission price -15%.



## Support

Contact the authors.

## Authors and acknowledgment
* Jussi Ikäheimo - model development, time series, testing
* Tomi J. Lindroos - time series, testing
* Anu Purhonen - time series
* Miika Rämä - district heating data
* Eric Harrison - testing

## License
To be specified.

