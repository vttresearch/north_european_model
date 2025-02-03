# North European energy system model

This repository contains North European energy system model. The model is built for the Backbone modelling framework. 

This readme has the following main sections
- [Installing Backbone and the North European Model](#installing-backbone-and-the-north-european-model)
- [Installing Julia, Conda, and setting up the environments](#installing-julia-conda-and-setting-up-the-environments)
- [Downloading required time series files](#downloading-required-time-series-files)
- [Building and copying input files for Backbone](#building-and-copying-input-files-for-backbone)
- [Running Backbone](#running-backbone)

## Installing Backbone and the North European Model

Make sure you have Backbone installed. For the moment, the North European Model works only in the Backbone master branch. 

If you do not have Backbone, use git to clone the Backbone from [release-3.x](https://gitlab.vtt.fi/backbone/backbone/). Master is the default branch, but you can use git to [switch between braches](https://gitlab.vtt.fi/backbone/backbone/-/wikis/Versions).

Use git to clone the repository of North European model. The easiest approach is to install North European model under Backbone, e.g. c:\backbone\north_european_model. The rest of the instructions are written assuming that the North European Model is installed under Backbone.


## Installing Julia, Conda, and setting up the environments

Make sure you have [Julia](https://julialang.org/)  version >= 1.5 installed. Install the Julia dependencies by starting Julia interactive session (e.g. typing 'Julia' in command prompt) in **backbone/north_european_model/** folder and running following commands:

	using Pkg
	Pkg.activate(".") 
    Pkg.instantiate()

Note: Building PyCall might be slow, but do not interrupt it. 

> _Tips for Windows users:_  You can navigate in the command prompt from one folder to another using `cd` commands, for example, `cd c:\backbone\north_european_model`. Instead of using the command prompt, you can also start Julia REPL and type, for example, `cd("c:/backbone/north_european_model")` to move to the right folder and continue by running the commands starting from `using Pkg`. As a third alternative, type `cmd` and press Enter in File Explorer when you are in your North European Model folder, and you don't need to move between folders before starting Julia interactive session. You can also start Julia REPL directly in the right folder by typing `julia` in File Explorer when you are in your North European Model folder. Note also that in some cases it may help to run Julia using admin rights.

The recommend approach to install Python dependencies is to set up a new environment in [Anaconda](https://www.anaconda.com/products/distribution). New users might want to use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) instead of more complicated Anaconda. 

Open the installed Anaconda Prompt, go to folder **backbone/north_european_model/timeseries/** (type, for example, `cd c:\backbone\north_european_model\timeseries`), and set up the environment by running the following commands

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

**VRE time series** are from ENTSO-E PECD dataset ([10.5281/zenodo](https://doi.org/10.5281/zenodo.3702418)). Following files are large and should be added in the **backbone/north_european_model/input/vre/** folder manually: 
* PECD-MAF2019-wide-PV.csv
* PECD-MAF2019-wide-WindOffshore.csv
* PECD-MAF2019-wide-WindOnshore.csv

Note: that these are older results with very low capacity factors. There is an option to use newer results but the processing of the input files is not yet included in this repository.

**Other time series data** are shared in this repository or queried from the ENTSO-E transparency platform. 

* Open Anaconda Prompt and activate the northEuropeanModel environment (`conda activate northEuropeanModel`)
* Go to folder **backbone\north_european_model\timeseries\Basic_processing** (for example, `cd c:\backbone\north_european_model\timeseries\Basic_processing`)
* Run **start_inputdata.py** (`python start_inputdata.py`) to create the uniform timeseries-files from the original input data. The run time is from 20 to 40 min and requires internet connection.
* Run **copyIntermediateCSVs.py** (`python copyIntermediateCSVs.py`) to copy created intermediate files to **backbone\north_european_model\timeseries\input** folder. Note that these are not model input, but intermediate storages for downloaded data.

If needed, each subfolder in backbone\north_european_model\timeseries\basic_processing\ contains a separate readme file explaining the data processing and requirements in greater detail.

Note: If you receive errors from missing packages, API key, or other, see previous section "Installing Julia, Conda, and setting up the environments".

Note: Basic processing has EV folder, but that is still work in progress and cannot be run with the currently shared files.


## Building and copying input files for Backbone

The package contains functions for building 
* the main excel input file for Backbone containing the energy system description (Julia), 
* VRE time series csv files (Julia),
* other time series csv files (Python)

Time series are built in two steps: for VRE (wind, solar) and others (electricity, hydro, and district heat). 

### Running julia for bb-input and VRE timeseries

The Julia script 'starthere.jl' in backbone\north_european_model folder runs three main functions 'convert_entsoe', 'convert_vre', and 'make_other_demands'. Check that all of these are active at the end of the file (not commented out).

Run the script **starthere.jl**, e.g. by typing `julia` in command prompt in **backbone\north_european_model** folder and running the following commands:

	using Pkg
	Pkg.activate(".") 
    Pkg.instantiate()
    include("starthere.jl")
	
> _Tips for Windows users:_ For running Julia in a specific folder, see tips in [Installing Julia, Conda, and setting up the environments](#installing-julia-conda-and-setting-up-the-environments).

In certain cases, Julia's PyCall might use different python version that does not find all packages. If julia crashes complaining, e.g. that pandas is not found, run 

	using PyCall
	pyimport_conda("pandas", "pandas")

And then rerun starthere.jl:

    include("starthere.jl")

By default, the output files in backbone\north_european_model\output will be 
* bb_input1-3x.xlsx
* bb_ts_cf_io.csv
* bb_ts_cf_io_10p.csv
* bb_ts_cf_io_50p.csv
* bb_ts_cf_io_90p.csv
* bb_ts_influx_other.csv

Copy these files to **backbone\input** folder.


### Running python for other timeseries

Open Anaconda Prompt and activate the northEuropeModel environment (`conda activate northEuropeanModel`). Then

* Go to folder **backbone\north_european_model\timeseries** (for example, `cd c:\backbone\north_european_model\timeseries`)
* Run **start_modelform.py** (`python start_modelform.py`) to convert intermediate timeseries files to the correct model format. 
* Output csv files are printed to **backbone\north_european_model\timeseries\output** folder. Copy these files to **backbone\input**

At the time of writing, the total size of files from Julia and Python are slightly below 2 Gb. 


### Manual additions

While we are working on the input data to be comprehensive, there are still some data that you need to add manually. It can be done in one of the two following ways.

Option 1. Copy bb_input_addData1.xlsx file from .\north_european_model\manual_additions\ to backbone\input folder.

Option 2. Copy data present in .\north_european_model\manual_additions\additional backbone input.xlsx to corresponding sheets of backbone\bb_input1-3x.xlsx file. Each data row of additional backbone input.xlsx includes 'replace' or 'add' indicating whether user should edit an existing value or add a new row to data tables.

Note that neither bb_input_addData1.xlsx nor additional backbone input.xlsx specifies scenario, which has to be determined by the data present. Currently bb_input_addData1.xlsx has data needed for "H2 heavy" scenario and \additional backbone input.xlsx for other scenarios.

Option 1 enables adding scenarios, modifying input data without the need to alter the automatically generated file, and saving your own modifications while generating a new base input data file. bb_input_addData1.xlsx has a mandatory file structure containing all tables of a full input file (i.e., bb_input_addData1.xlsx) except for ts_cf, ts_inlux, and ts_node.


### Copying run specification files


A working Backbone model requires certain run specification files for GAMS

* Copy **cplex.opt** from backbone\north_european_model\GAMS_files to **backbone** root folder. 
* Use a suitable **modelsInit.gms** and **scheduleInit.gms** file, for example, copy **modelsInit_example.gms** and **scheduleInit_example.gms** from backbone\north_european_model\GAMS_files to **backbone\input** folder and rename them to modelsInit.gms and scheduleInit.gms. 
* Copy **1_options.gms**, **changes.inc** and **timeandsamples.inc** from backbone\north_european_model\GAMS_files to **backbone\input** folder. (Note: If 1_options.gms is missing, copy temp_1_options.gms from backbone\inputTemplates to backbone\input and rename it to 1_options.gms.)

Note: Included scheduleInit.gms file has a specific structure so that it works with *tsYear* and *modelledDays* parameters. If using your own file, adapt a similar structure to the file.

Note: changes.inc does quite many different tasks to read and process input data, see the content and introduction at the beginning of the file. The file has a separate section where user can do additional changes.

Note: changes.inc calls csv2gdx and some older versions of csv2gdx do not support very long time series. In this case, install also a more recent gams and manually add a hard coded file path to changes inc, e.g. converting `$call 'csv2gdx ...` to `$call 'c:\GAMS\45\csv2gdx ...`


## Running Backbone

Run the model by running Backbone.gms in GAMS. The model requires the following command line option (use two hyphens in the beginning)

* **--input_file_excel** is a mandatory parameter defining the used input excel file name (e.g. bb_input1-3x.xlsx)

The model supports a few user given additional options. All these are optional. Behaviour and default values are listed below (use two hyphens in the beginning of each).

* --modelYear [2025, 2030]. Default 2025. This parameter currently two options and impacts only district heating demand. Generation and transfer capacities for different model years are changed in the starthere.jl
* --tsYear [0, 2011-2016]. Default 2015. This parameter allows a quick selection of which time series year the model uses for profiles and annual demands and water inflows. Giving this parameter greatly reduces the solve time as the model drops ts (time series) data from other years and loops the selected time series year. By giving value 0, user can run the model with multiyear time series, but the user is responsible for giving the correct starting time step and checking for error. This feature (tsYear=0) is untested.
* --modelledDays [1-365]. Default 365. This option defines the amount of modelled days. If used with tsYear, the maximum value is 365. Otherwise user can give longer time periods, but must check that original timeseries length will not be exceeded.
* --forecasts [1, 2, 4]. Default 4. Activates forecasts in the model and requires 10p, 50p, and 90p time series filen in the input folder. Currently accepted values are 1 (realized values only), 2 (realized values and 1 central forecast), or 4 (realized values, 1 central forecast, 1 difficult forecast, 1 easy forecast). It is recommended to use 4 forecasts due to improved hydro power modelling. 

Working command line options for backbone.gms would be, for example:

	--input_file_excel=bb_input1-3x.xlsx --tsYear=2016 --modelledDays=7


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

