# North European energy system model

This repository contains North European energy system model. The model is built for the Backbone modelling framework. 

This readme has the following main sections
- [Installing Backbone and the North European Model](#installing-backbone-and-the-north-european-model)
- [Updating Backbone North European Model](#updating-backbone-and-the-north-european-model)
- [Installing MiniConda and setting up the environments](#installing-miniconda-and-setting-up-the-environment)
- [Downloading required time series files](#downloading-required-time-series-files)
- [Building and copying input files for Backbone](#building-and-copying-input-files-for-backbone)
- [Running Backbone](#running-backbone)


## Authors and acknowledgment
* Jussi Ikäheimo - model development, time series, testing
* Tomi J. Lindroos - model development, time series, testing
* Anu Purhonen - time series
* Miika Rämä - district heating data
* Pauli Hiltinen - district heating data
* Eric Harrison - testing


## License

This work is licenced under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

https://creativecommons.org/licenses/by-nc-sa/4.0/ 


## Support

Contact the authors.



## Installing Backbone and the North European Model

**CHECKPOINT**: Install [tortoiseGit](https://tortoisegit.org/docs/tortoisegit/tgit-intro-install.html) if not yet installed

**CHECKPOINT**: You should have a Backbone installed to e.g. c:\backbone. If not, first install the Backbone master branch. See installation instructions from https://gitlab.vtt.fi/backbone/backbone/-/wikis/home and then proceed to following steps to install the Northern European model. 

**Installing with TortoiseGit**
 * Open file browser and go to the Backbone folder
 * Create a new folder "north_european_model" under the backbone folder (c:\backbone\north_european_model).
 * Right click "north_european_model" folder and select "clone" under the tortoise git selection pane
 * copy "https://github.com/vttresearch/north_european_model" to **URL**
 * double check that the installation **Directory** is `c:\backbone\north_european_model` and not c:\backbone\north_european_model\north_european_model which tortoiseGit might suggest

**Switch to timeseries_update branch**
 * Right click "north_european_model" folder and select "Switch/Checkout" under the tortoise git selection pane
 * select timeseries_update from available branches, click ok
 * Right click "north_european_model" folder and select "pull" under the tortoise git selection pane, click ok

For the moment, the updated North European Model works in timeseries_update and only with the Backbone master branch. Pulling after switching the branch guarantees that you have the latest version.

The rest of the instructions are written assuming that the North European Model is installed under Backbone to `c:\backbone\north_european_model`, but both Backbone and Northern European Model of course support also other installation directories. 


## Updating Backbone and the North European Model

For the moment, the updated North European Model works in timeseries_update and only with the Backbone master branch. 

**Check that you are in correct backbone branch**
 * Right click "backbone" folder and select "Switch/Checkout" from tortoiseGit. This shows the current branch. 
 * Switch to master and pull new version

**Check that you are in correct North European Model branch**
 * Right click "north_european_model" folder and select "Switch/Checkout" from tortoiseGit. This shows the current branch. 
 * Switch to timeseries_update and pull new version

**Note:** if you have edited any of the git files, switchin and pullin will cause an error. In these cases you must revert all changes before.
 * right click the folder and select "Revert" from tortoiseGit. 
 * Check the file list and decide if you need backups from those files or not
 * revert all changes


## Installing Miniconda and setting up the environment

[Back to top](#North-European-energy-system-model)

**CHECKPOINT**: Install [miniConda](https://www.anaconda.com/docs/getting-started/miniconda/install) if not yet installed. 

These instructions are written for miniconda, but users can of course choose other conda versions as well.
  * Open the installed Miniconda Prompt (e.g. type miniconda to windows search bar), 
  * In the miniconda, go to folder **backbone/north_european_model/** by typing two commands: `c:` and then `cd c:\backbone\north_european_model`
  * In the miniconda, set up the environment by running the following commands

`   conda env create -f environment.yml
    conda activate northEuropeanModel`

Installed environment needs few additional packages as conda does not automatically find them. After creating and activating the northEuropeanModel environment, install following additional packages in the miniconda by typing:

    pip install gdxpds
	pip install gamsapi[transfer]==xx.y.z

where xx.y.z is your GAMS version. You can find the correct version by opening gams Studio, clicking help -> GAMS Licensing -> check GAMS Distribution xx.y.z

After these steps, you should have required softwares and environment ready. 

NOTE: GamsAPI is possible to install also for much older GAMS versions, see https://github.com/NREL/gdx-pandas

 


## Downloading required time series files

[Back to top](#North-European-energy-system-model)

The North Europe model has some time series source files that are too large to be shared in this repository. Following time series should be prepared
* **Electricity demand profiles**
	* Download [Demand-Profiles.zip](https://2024.entsos-tyndp-scenarios.eu/wp-content/uploads/2024/draft2024-input-output/Demand-Profiles.zip) from ENTSO-E TYNDP 2024 scenarios
	* extract following two files from the zip: "Demand Profiles\NT\Electricity demand profiles\2030_National Trends.xlsx", and "Demand Profiles\NT\Electricity demand profiles\2040_National Trends.xlsx"
	* copy file to to `c:/backbone/north_european_model/src_files/timeseries`
	* rename them to elec_2030_National_Trends.xlsx, and elec_2040_National_Trends.xlsx
* **VRE time series** are from ENTSO-E PECD dataset ([10.5281/zenodo](https://doi.org/10.5281/zenodo.3702418)). Following files should be copied to the `c:/backbone/north_european_model/src_files/timeseries/` folder: 
	* PECD-MAF2019-wide-PV.csv
	* PECD-MAF2019-wide-WindOffshore.csv
	* PECD-MAF2019-wide-WindOnshore.csv

Other time series data (Hydro, District heating, hydrogen, industry) are shared in this repository 

Note: EV timeseries are still work-in-progress, but will be added 


## Building and copying input files for Backbone

[Back to top](#North-European-energy-system-model)

### Building input files

Inputs are build with python script which is easiest to run in the miniconda handling the packages and environments
 * Open the installed Miniconda Prompt (e.g. type miniconda to windows search bar), 
 * In the miniconda, go to folder **backbone/north_european_model/** by typing two commands: `c:` and then `cd c:\backbone\north_european_model`
 * In the miniconda, activate the northEuropeanModel environment by typing `conda activate northEuropeanModel`
 * In the miniconda, Run **build_input_data.py** by typing (`python build_input_data.py input_folder=src_files config_file=config_NT2025.ini`) 
 * Output files are written to **'backbone\north_european_model\input_National Trends_2025\'** folder. Copy these files to **backbone\input**

At the time of writing, the created "National Trends" takes about 500 Mb, is generated in ~12 minutes, and has ~300 files. Writing some larger sets of GDX files might take 60-80secs and the code might seem stuck for those periods, but should eventually proceed.

Python functions to build input data is called with syntax `python build_input_data.py input_folder=<directory> config_file=<filename>` where
	* input_folder is the directory for excel data, large timeseries files, and GAMS file templates. In repository the default folder is **src_files**.
	* config_file is a list of instruction to generate the data for the scenario. The repository currently shares following config files:
		* config_NT2025.ini for **National Trends** scenario
		* confic_test.ini for faster testing of the model
		* H2 heavy will be added soon

Users can create their own config files and store them locally. Editing any of the files in git will cause version control issues with git and is not recommended.

Processed input files are written to `c:\Backbone\north_european_model\<output_folder>`, where
	* output folder is a combination of <output_folder_prefix>_<scenario>_<year>_<alternative> defined in the called config file


### Checking run specification files

The python script constructs following files
* import_timeseries.inc - this is a specific file containing instructions for Backbone about how to import timeseries GDX files

In addition, the script automatically copies following run specification files from `src_files\GAMS_files` to the `<output_folder>` and user is free to edit them afterwards. In most cases, users do not need to edit these at all.
* 1_options.gms - some solver settings documented inside the file
* timeAndSamples.inc - sets defining timestep and forecast domains in Backbone 
* modelsInit_example.gms - a default modelsInit file calling scheduleInit.gms
* scheduleInit.gms - a tailored scheduleInit file for the Northern European Backbone
* changes.inc - reads possible additional excel data, reads timeseries gdx files, and allows users to add their own project specific changes to the end of the file

Note: Included scheduleInit.gms and changes.inc files have a specific structure so that it works with *climateYear* and *modelledDays* parameters. If using your own files, adapt a similar structure to the file.


### Copying input files to c:\backbone\input

The default use case often is the copy the full content of `<output_folder>` to c:\backbone\input and run the constructed model from there.

Alternative approach is to run the model directly from `<output_folder>` by giving `--input_folder='.\north_european_model\<output_folder>'` command line option for the Backbone.



## Running Backbone

[Back to top](#North-European-energy-system-model)

Run the model by running Backbone.gms in GAMS. The model supports the following command line options (use two hyphens in the beginning)

* **--input_file_excel** is a mandatory parameter for both defining the used input excel file name (e.g. inputData.xlsx)
* --climateYear [0, 1982-2016]. Default 2015. This parameter allows a quick selection of which time series year the model uses for profiles and annual demands and water inflows. Giving this parameter greatly reduces the solve time as the model drops ts (time series) data from other years and loops the selected time series year. By giving value 0, user can run the model with multiyear time series, but the user is responsible for giving the correct starting time step and checking for error. This feature (tsYear=0) is untested.
* --modelledDays [1-365]. Default 365. This option defines the amount of modelled days. If used with tsYear, the maximum value is 365. Otherwise user can give longer time periods, but must check that original timeseries length will not be exceeded.
* --forecasts [1, 2, 4]. Default 4. Activates forecasts in the model and requires 10p, 50p, and 90p time series filen in the input folder. Currently accepted values are 1 (realized values only), 2 (realized values and 1 central forecast), or 4 (realized values, 1 central forecast, 1 difficult forecast, 1 easy forecast). It is recommended to use 4 forecasts due to improved hydro power modelling.
* --input_dir=directory. Default 'input'. Allows custom locations of input directory


Working command line options for backbone.gms would be, for example:

	Running the model with all default assumptions
	_--input_file_excel=inputData.xlsx_  	
	running the selected climate year, 1 week test
	_--input_file_excel=inputData.xlsx --modelledDays=7 --climateYear=1995_
	running the model directly from <output> folder
	_--input_folder='.\north_european_model\input_National Trends_2025' --input_file_excel=inputData.xlsx_  

Results are written to c:\backbone\output\results.gdx

