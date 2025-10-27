# North European energy system model

This repository contains the North European energy system model. The model is built for the Backbone modelling framework. 

This readme has the following main sections
- [Installing Backbone and the North European Model](#installing-backbone-and-the-north-european-model)
- [Updating Backbone North European Model](#updating-backbone-and-the-north-european-model)
- [Installing MiniConda and setting up the environments](#installing-miniconda-and-setting-up-the-environment)
- [Downloading required time series files](#downloading-required-time-series-files)
- [Building input files for Backbone and running the model](#Building-input-files-for-Backbone-and-running-the-model)
- [Running Backbone](#running-backbone)


## Authors and acknowledgments
* Jussi Ikäheimo - Model development, time series, testing
* Tomi J. Lindroos - Model development, time series, testing
* Anu Purhonen - Time series
* Miika Rämä - District heating data
* Pauli Hiltunen - District heating data, testing
* Eric Harrison - Data sets, testing


## License

This work is licenced under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license.

https://creativecommons.org/licenses/by-nc-sa/4.0/ 


## Citation

Ikäheimo, J., Lindroos, T.J., Purhonen, A., Rämä, M., Hiltunen, P., and Harrison, E. North European energy system model. https://github.com/vttresearch/north_european_model


## Support

Contact the authors.


## Installing Backbone and the North European Model

**CHECKPOINT**: Install [tortoiseGit](https://tortoisegit.org/docs/tortoisegit/tgit-intro-install.html) if not yet installed.

**CHECKPOINT**: You should have Backbone installed to e.g. c:\backbone. If not, first install the Backbone `master` branch (not e.g. the `release-3.x` branch). See installation instructions from https://gitlab.vtt.fi/backbone/backbone/-/wikis/home and then proceed to the following steps to install the North European model. 

The rest of the instructions are written assuming that the North European Model is installed under Backbone to `c:\backbone\north_european_model`, but both Backbone and North European Model of course support also other installation directories.

**Installing with TortoiseGit**
 * Open a file browser and go to the Backbone folder.
 * Create a new folder "north_european_model" under the backbone folder (c:\backbone\north_european_model).
 * Right click the `north_european_model` folder and select "clone" under the tortoise git selection pane.
 * Copy "https://github.com/vttresearch/north_european_model" to **URL**.
 * Double check that the installation **Directory** is `c:\backbone\north_european_model` and not `c:\backbone\north_european_model\north_european_model` which tortoiseGit might suggest.


## Updating Backbone and the North European Model

For the moment, the North European Model works only with the Backbone `master` branch. 

**Check that you are in correct backbone branch**
 * Right click "backbone" folder and select "Switch/Checkout" from tortoiseGit. This shows the current branch. 
 * Switch to `master` and pull the new version.

**Check that you are in correct North European Model branch**
 * Right click "north_european_model" folder and "Pull" new version with the TortoiseGit. 
 * In case, you are still in "timeseries_update" branch, switch to `main` and pull the new version.

**Note:** if you have edited any of the git files, switching and pulling will cause an error. In these cases you must revert all changes before.
 * Right click the folder and select "Revert" from tortoiseGit. 
 * Check the file list and decide if you need backups from those files or not.
 * Revert all changes.


## Installing Miniconda and setting up the environment

[Back to top](#North-European-energy-system-model)

**CHECKPOINT**: Install [miniConda](https://www.anaconda.com/docs/getting-started/miniconda/install) if not yet installed. 

These instructions are written for Miniconda, but users can of course choose other conda versions as well.
  * Open the installed Miniconda Prompt (e.g. type `miniconda` or `anaconda` to windows search bar), 
  * In Miniconda, go to folder **backbone/north_european_model/** by typing two commands: `c:` and then `cd c:\backbone\north_european_model`.
  * In Miniconda, set up the environment by running the following commands:
	
    ```
	conda env create -f environment.yml
	conda activate northEuropeanModel
	```
	

The installed environment needs a few additional packages as Miniconda does not automatically find them. After creating and activating the `northEuropeanModel` environment, install the following additional packages in Miniconda by typing:

```
pip install gdxpds
pip install gamsapi[transfer]==xx.y.z
```

where xx.y.z is your GAMS version. You can find the correct version by opening GAMS Studio, clicking **Help** -> **GAMS Licensing** -> check GAMS Distribution xx.y.z.

After these steps, you should have the required software and environment ready. 

NOTE: GamsAPI is possible to install also for much older GAMS versions, see https://github.com/NREL/gdx-pandas

 


## Downloading required time series files

[Back to top](#North-European-energy-system-model)

The North European model has some time series source files that are too large to be shared in this repository. The following time series should be prepared:
* **Electricity demand profiles**
	* Download [Demand-Profiles.zip](https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-inputs/Demand-Profiles.zip) from ENTSO-E TYNDP 2024 scenarios. If the link is broken, try "demand profiles" from https://2024.entsos-tyndp-scenarios.eu/download/.
	* Extract the following two files from the zip:
		* `Demand Profiles\NT\Electricity demand profiles\2030_National Trends.xlsx`
		* `Demand Profiles\NT\Electricity demand profiles\2040_National Trends.xlsx`
	* Copy the files to `c:/backbone/north_european_model/src_files/timeseries`.
	* Rename them to `elec_2030_National_Trends.xlsx`, and `elec_2040_National_Trends.xlsx` (note the underscore in "National_Trends").
* **VRE time series from MAF2019** are from an older ENTSO-E PECD dataset ([10.5281/zenodo](https://doi.org/10.5281/zenodo.3702418)). Copy the following files to `c:/backbone/north_european_model/src_files/timeseries/` folder: 
	* `PECD-MAF2019-wide-PV.csv`
	* `PECD-MAF2019-wide-WindOffshore.csv`
	* `PECD-MAF2019-wide-WindOnshore.csv`
* **The new, updated VRE time series from PECD** are from 2025 ([PECD database](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd?tab=download)). 
	* download timeseries from PECD portal with you preferred settings, e.g. 
		* PV:
			* Temporal period - historical
			* Origin - ERA5 reanalysis
			* Variable - Energy - Solar Energy - Solar generation capacity factor
			* Spatial resolution - Region aggregated timeseries - SZON (Onshore bidding zones)
		* Onshore:
			* Temporal period - historical
			* Origin - ERA5 reanalysis
			* Variable - Energy - Wind Energy - Wind power onshore capacity factor
			* Technological specification - onshore wind turbine - 30 (Existing technologies)
			* Spatial resolution - Region aggregated timeseries - PEON (Pan-European Onshore Zones)
		* Offshore
			* Temporal period - historical
			* Origin - ERA5 reanalysis
			* Variable - Energy - Wind Energy - Wind power offshore capacity factor
			* Technological specification - offshore wind turbine - 20 (Existing technologies)
			* Spatial resolution - Region aggregated timeseries - PEOF (Pan-European Offshore Zones)
		* Note: max 20 years can be downloaded at once. The full dataset (PV, onshore, offshore from 1982 to 2016) needs 6 downloads. Other time series limit the years to 1982-2016.
	* Create the following three folders: `c:/backbone/north_european_model/src_files/timeseries/PECD-PV`, `timeseries/PECD-onshore`, and `timeseries/PECD-offshore` 
	* Copy the timeseries csv files to these folders.

See [Choosing VRE processor](#Choosing-VRE-processor) for how you can choose which VRE datasets to use.

Other time series data (Hydro, District heating, hydrogen, industry) are shared in this repository and do not yet have alternative data sources.

Note: EV timeseries are still work-in-progress, but will be added.

## (Optional) Installing and setting up a Spine Toolbox project

[Spine Toolbox](https://github.com/spine-tools/Spine-Toolbox) is an open source Python package to manage data, scenarios and workflows for modelling and simulation.
You can use the Spine Toolbox for the workflow management if you so choose.

First install Spine Toolbox:

1. Install pipx. pipx helps in creating an isolated environment for Spine Toolbox to avoid package conflicts. Open a terminal and run

```
python -m pip install --user pipx
```

2. After pipx has been installed, run

```
python -m pipx ensurepath
```

3. Restart the terminal or re-login for the changes of the latest command to take effect.

4. Choose which Spine Toolbox version to install. Latest release version from PyPi is installed using

```
python -m pipx install spinetoolbox
```

Open Spine Toolbox by typing in the terminal: 

```
spinetoolbox
```

Go to:
File -> Open project -> choose the north_european_model folder. You can see a Spine Toolbox logo next to it. If your North European Model is installed under Backbone, it is not enough to just choose the Backbone folder as it is a separate Spine Toolbox project.

To get the Miniconda environment 'northEuropeanModel' running in Spine Toolbox, it needs to be set as the Python kernel:

1. Open Miniconda Prompt.

2. Activate the `northEuropeanModel` environment by typing `conda activate northEuropeanModel`.
   
3. Install an additional package by typing `pip install ipykernel`.

4. In Spine Toolbox, double-click the `build_input_data` project item in the Design View, which will open the Tool specification editor.

5. In the Tool specification editor, select Jupyter Console and then, next to `Kernel`, select the northEuropeanModel environment.

Take a copy of the BB_data_template.sqlite database from the Backbone folder. It is located in
**backbone/.spinetoolbox/items/bb_data_template**
You can put it anywhere you like and rename it if you wish.
Go to the Spine Toolbox Design View and click the Input_data project item. Choose the path to that copied database file from the Data Store Properties window that opened to the right side.

## Building input files for Backbone and running the model

[Back to top](#North-European-energy-system-model)

### Building input files

Inputs are build with a python script which is easiest to run with Miniconda handling the packages and environments.
 * Open the installed Miniconda Prompt (e.g. type `miniconda` to Windows search bar), 
 * In Miniconda, go to the model folder e.g. `c:\backbone\north_european_model\` by typing two commands: `c:` and then `cd c:\backbone\north_european_model`
 * In Miniconda, activate the `northEuropeanModel` environment by typing `conda activate northEuropeanModel`.
 * In Miniconda, run `build_input_data.py` by typing (`python build_input_data.py src_files config_NT2030.ini`).


At the time of writing, the created "National Trends" takes about 500 Mb, is generated in ~7 minutes, and has ~300 files. Writing some larger sets of GDX files might take up to 60 seconds and the code might seem stuck for those periods, but should eventually proceed.

The `config_NT2030.ini` writes output files to **'backbone\north_european_model\input_National Trends_2030\'** folder. 


You can run Backbone either directly from the created output folder or by copying these files to **backbone\input**, see instructions from [Running Backbone](#running-backbone).


### Choosing VRE processor

Current `config_NT2030.ini` is using PECD timeseries, but old MAF2019 processor is still available. It is not recommended to edit config files stored in GIT, but instead take a copy, rename it, and edit your own file.

Timeseries processors are selected and configured in the `timeseries_specs = {}` dictionary in config files. The default configuration for new PECD processors for onshore wind looks like this:

	'wind_onshore': {
		'processor_name': 'VRE_PECD',
		'bb_parameter': 'ts_cf',
		'bb_parameter_dimensions': ['flow', 'node', 'f', 't'],
		'custom_column_value': {'flow': 'onshore'},
		'gdx_name_suffix': 'wind_onshore',
		'calculate_average_year': True,
		'rounding_precision': 5,
		'input_file': 'PECD-onshore/',   # folder, not file
		'attached_grid': 'elec'
	},


and the old configuration for the MAF2019 processor for PV would like this:

	'PV': {
		'processor_name': 'VRE_MAF2019',
		'bb_parameter': 'ts_cf',
  		'bb_parameter_dimensions': ['flow', 'node', 'f', 't'],
		'custom_column_value': {'flow': 'PV'},
		'gdx_name_suffix': 'PV',
		'calculate_average_year': True,
		'rounding_precision': 5,
		'input_file': 'PECD-MAF2019-wide-PV.csv',
		'attached_grid': 'elec'
	},



### Copying input files to c:\backbone\input

A recommended approach is to run the model directly from `<output_folder>` by giving the `--input_folder='.\north_european_model\<output_folder>'` command line option for Backbone.

Alternative use cases are to rename the folder to avoid a case where the consecutive runs of the python script would overwrite the files, or to copy the full content of `<output_folder>` to c:\backbone\input and run the constructed model from there.



### Checking run specification files

Users might want the check the contents of following files, but this is not needed if the default settings are ok.

The script automatically copies the following required run specification files from `src_files\GAMS_files` to `<output_folder>`, and the user is free to edit them afterwards. In most cases, users do not need to edit these at all.
* `1_options.gms` - some solver settings documented inside the file
* `timeAndSamples.inc` - sets defining timestep and forecast domains in Backbone 
* `modelsInit_example.gms` - a default modelsInit file calling scheduleInit.gms
* `scheduleInit.gms` - a tailored scheduleInit file for the Northern European Backbone
* `changes.inc` - reads possible additional excel data, reads timeseries gdx files, and allows users to add their own project specific changes to the end of the file

The python script constructs following files
* `import_timeseries.inc` - this is a specific file containing instructions for Backbone about how to import timeseries GDX files

**Note:** The included `scheduleInit.gms` and `changes.inc` files have a specific structure to make them work with *climateYear* and *modelledDays* parameters. If using your own files, adapt a similar structure to them.



### Building own config files

Users can create their own config files and store them locally. Editing any of the files in git will cause version control issues with git and is not recommended.

Python functions to build input data is called with syntax `python build_input_data.py <input_folder> <config_file_name>` where
 * `input_folder` is the directory for Excel data, large timeseries files, and GAMS file templates. In the repository, the default folder is `src_files`.
 * `config_file_name` is a list of instruction to generate the data for the scenario. The repository currently shares following config files:
    * `config_NT2025.ini` for the **National Trends** scenario.
	* `config_test.ini` for faster testing of the model.
	* H2 heavy will be added soon.

Processed input files are written to `c:\Backbone\north_european_model\<output_folder>`, where the output folder name is a combination of `<output_folder_prefix>_<scenario>_<year>_<alternative>`, defined in the called config file.




## Running Backbone

[Back to top](#North-European-energy-system-model)

Run the model by running Backbone.gms in GAMS. The model supports the following command line options (use two hyphens in the beginning)

* `--input_file_excel` is a mandatory parameter for defining the used input Excel file name (e.g. inputData.xlsx)
* `--climateYear` [0, 1982-2016]. Default 2015. This parameter allows a quick selection of which time series year the model uses for profiles and annual demands and water inflows. Giving this parameter greatly reduces the solve time as the model drops ts (time series) data from other years and loops the selected time series year. By giving value 0, user can run the model with multiyear time series, but the user is responsible for giving the correct starting time step and checking for error. This feature (tsYear=0) is untested.
* `--modelledDays` [1-365]. Default 365. This option defines the amount of modelled days. If used with tsYear, the maximum value is 365. Otherwise user can give longer time periods, but must check that original timeseries length will not be exceeded.
* `--forecasts` [1, 2, 4]. Default 4. Activates forecasts in the model and requires 10p, 50p, and 90p time series filen in the input folder. Currently accepted values are 1 (realized values only), 2 (realized values and 1 central forecast), or 4 (realized values, 1 central forecast, 1 difficult forecast, 1 easy forecast). It is recommended to use 4 forecasts due to improved hydro power modelling.
* `--input_dir` allows setting a custom location for the input directory. The default value is 'input'. 

Working command line options for `backbone.gms` would be, for example:
* running the model directly from <output_folder>, full year, climate year 2011: `--input_dir=".\north_european_model\input_National Trends_2030" --input_file_excel=inputData.xlsx --climateYear=1995`
* Running the model from `.\backbone\input` with all default assumptions: `--input_file_excel=inputData.xlsx`
* running the selected climate year, 1 week test: `--input_file_excel=inputData.xlsx --modelledDays=7 --climateYear=1995`


**NOTE:** Use " instead of ' when writing e.g. folder names with spaces. For example, --input_dir='.\dir with spaces' does not work in many workflows, but --input_dir=".\dir with spaces" should work.

Results from the model run are written to `c:\backbone\output\results.gdx` unless the destination is modified by some option or workflow manager, such as Spine Toolbox.

