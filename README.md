# North European energy system model

This repository contains North European energy system model. The model is built for the Backbone modelling framework. 

This readme has the following main sections
- [Installing Backbone and the North European Model](#installing-backbone-and-the-north-european-model)
- [Installing Julia, Conda, and setting up the environments](#installing-julia-conda-and-setting-up-the-environments)
- [Downloading required time series files](#downloading-required-time-series-files)
- [Building and copying input files for Backbone](#building-and-copying-input-files-for-backbone)
- [Running Backbone](#running-backbone)


## Authors and acknowledgment
* Jussi Ikäheimo - model development, time series, testing
* Tomi J. Lindroos - model development, time series, testing
* Anu Purhonen - time series
* Miika Rämä - district heating data
* Eric Harrison - testing


## License

This work is licenced under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

https://creativecommons.org/licenses/by-nc-sa/4.0/ 


## Support

Contact the authors.



## Installing Backbone and the North European Model

Make sure you have Backbone installed. For the moment, the North European Model works only in the Backbone master branch. 

If you do not have Backbone, use git to clone the Backbone from [master](https://gitlab.vtt.fi/backbone/backbone/). Master is the default branch.

Use git to clone the repository of North European model. The rest of the instructions are written assuming that the North European Model is installed under Backbone to `c:\backbone\north_european_model`, but both Backbone and Northern European Model of course support also other installation directories. 


## Installing Miniconda and setting up the environment

** Python via Miniconda ** 

The recommend approach to install Python and related packages is to set up a new environment in Miniconda. 
  * Install the latest [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html). 
  * Open the installed Miniconda Prompt (e.g. type miniconda to windows search bar), 
  * go to folder **backbone/north_european_model/** (for example, type `cd c:\backbone\north_european_model`), and
  * set up the environment by running the following commands

`    conda env create -f environment.yml
    conda activate northEuropeanModel`

Installed environment needs few additional packages as conda does not automatically find them. After creating and activating the northEuropeanModel environment, install following additional packages by typing:

    pip install gdxpds
	pip install gamsapi[transfer]==xx.y.z

where xx.y.z is your GAMS version. You can find the correct version by opening gams Studio, clicking help -> GAMS Licensing -> check GAMS Distribution xx.y.z

After these steps, you should have required softwares and environment ready. 

NOTE: GamsAPI is possible to install also for much older GAMS versions, see https://github.com/NREL/gdx-pandas

Note: These instructions are written for miniconda, but users can of course choose other conda versions as well. 


## Downloading required time series files

[Back to top](#North-European-energy-system-model)

The North Europe model has some time series source files that are too large to be shared in this repository. Following time series should be prepared
* Electricity demand profiles
	* Download [Demand-Profiles.zip](https://2024.entsos-tyndp-scenarios.eu/wp-content/uploads/2024/draft2024-input-output/Demand-Profiles.zip) from ENTSO-E TYNDP 2024 scenarios
	* extract following two files from the zip: "Demand Profiles\NT\Electricity demand profiles\2030_National Trends.xlsx", and "Demand Profiles\NT\Electricity demand profiles\2040_National Trends.xlsx"
	* copy file to to "c:/backbone/north_european_model/inputFiles/timeseries\" 
	* rename them to elec_2030_National_Trends.xlsx, and elec_2040_National_Trends.xlsx
* VRE time series are from ENTSO-E PECD dataset ([10.5281/zenodo](https://doi.org/10.5281/zenodo.3702418)). Following files are large and should be added in the **c:/backbone/north_european_model/inputFiles/timeseries/** folder manually: 
	* PECD-MAF2019-wide-PV.csv
	* PECD-MAF2019-wide-WindOffshore.csv
	* PECD-MAF2019-wide-WindOnshore.csv

Other time series data (Hydro, District heating, hydrogen, industry) are shared in this repository 

Note: EV timeseries are still work-in-progress, but will be added 


## Building and copying input files for Backbone

Open Miniconda Prompt and activate the northEuropeModel environment (`conda activate northEuropeanModel`). Then

* Go to folder **backbone\north_european_model** (for example, `cd c:\backbone\north_european_model`)
* Run **build_input_data.py** by typing (`python build_input_data.py input_folder=src_files config_file=config_NT2025.ini`) 
* Output files are written to **'backbone\north_european_model\input_National Trends_2025\'** folder. Copy these files to **backbone\input**

At the time of writing, the total size of created files is about 500 Mb and the time of generating is around 8 minutes. Note: Writing large GDX files might take 20-30secs and the code might seem stuck for those periods.

Python functions to build input data is called as _python build_input_data.py input_folder=<directory> config_file=<filename>_
	* The source file directory in repository is **src_files**
	* The repository currently shares config file for **National Trends** scenario
	* H2 heavy will be added soon
* users can create their own conversions with the examples in config_NT2025.ini, config_H2heavy.ini (TBD)
* processed files are written to c:\Backbone\north_european_model\<output_folder>, where
	* output folder is a combination of <output_folder_prefix>_<scenario>_<year>_<alternative> defined in the called config file


### Copying run specification files


A working Backbone model requires certain run specification files for GAMS

* Use a suitable **modelsInit.gms** and **scheduleInit.gms** file, for example, copy **modelsInit_example.gms** and **scheduleInit_example.gms** from backbone\north_european_model\GAMS_files to **backbone\input** folder and rename them to modelsInit.gms and scheduleInit.gms. 
* Copy **1_options.gms**, **changes.inc** and **timeandsamples.inc** from backbone\north_european_model\GAMS_files to **backbone\input** folder. (Note: If 1_options.gms is missing, copy temp_1_options.gms from backbone\inputTemplates to backbone\input and rename it to 1_options.gms.)

Note: Included scheduleInit.gms file has a specific structure so that it works with *tsYear* and *modelledDays* parameters. If using your own file, adapt a similar structure to the file.

Note: changes.inc does additional processing of input data, see the content and introduction at the beginning of the file. User can add their own project specific changes to the end of the file




## Running Backbone

[Back to top](#North-European-energy-system-model)

Run the model by running Backbone.gms in GAMS. The model supports the following command line options (use two hyphens in the beginning)

* **--input_file_excel** is a mandatory parameter for both defining the used input excel file name (e.g. inputData.xlsx)
* --climateYear [0, 1982-2016]. Default 2015. This parameter allows a quick selection of which time series year the model uses for profiles and annual demands and water inflows. Giving this parameter greatly reduces the solve time as the model drops ts (time series) data from other years and loops the selected time series year. By giving value 0, user can run the model with multiyear time series, but the user is responsible for giving the correct starting time step and checking for error. This feature (tsYear=0) is untested.
* --modelledDays [1-365]. Default 365. This option defines the amount of modelled days. If used with tsYear, the maximum value is 365. Otherwise user can give longer time periods, but must check that original timeseries length will not be exceeded.
* --forecasts [1, 2, 4]. Default 4. Activates forecasts in the model and requires 10p, 50p, and 90p time series filen in the input folder. Currently accepted values are 1 (realized values only), 2 (realized values and 1 central forecast), or 4 (realized values, 1 central forecast, 1 difficult forecast, 1 easy forecast). It is recommended to use 4 forecasts due to improved hydro power modelling.
* --input_dir=directory. Default 'input'. Allows custom locations of input directory.


Working command line options for backbone.gms would be, for example:

	Running the model with all default assumptions
	--input_file_excel=inputData.xlsx   	
	running the selected climate year, 1 week test
	--input_file_excel=inputData.xlsx --modelledDays=7 --climateYear=1995     

The run will abort if following GDX files cannot be found from the input directory
* TBD

Results are written to c:\backbone\output\results.gdx

