# North European energy system model

This repository contains the North European energy system model. The model is built for the Backbone modelling framework. 

This readme has the following main sections
- [Documentation](#documentation)
- [Installing Backbone and the North European Model](#installing-backbone-and-the-north-european-model)
- [Updating Backbone North European Model](#updating-backbone-and-the-north-european-model)
- [Installing MiniConda and setting up the environments](#installing-miniconda-and-setting-up-the-environment)
- [Updating the conda environment](#Updating-the-conda-environment)
- [Downloading required time series files](#downloading-required-time-series-files)
- [Building input files for Backbone and running the model](#Building-input-files-for-Backbone-and-running-the-model)
- [Running Backbone](#running-backbone)
- [Tool assisted running](#tool-assisted-running)


## Authors and acknowledgments
* Tomi J. Lindroos - Model development, time series, testing
* Jussi Ikäheimo - Model development, time series, testing
* Anu Purhonen - Time series
* Miika Rämä - District heating data
* Pauli Hiltunen - District heating data, testing
* Eric Harrison - Data sets, testing
* Justinas Jasiūnas - H2 heavy scenario, testing
* Touko Kumpulainen - District heating data, testing


## License

Copyright (c) 2026 VTT Technical Research Centre of Finland Ltd

This work is licensed under a Creative Commons Attribution 4.0 International
(CC BY 4.0) license. See [LICENSE](LICENSE) for the full terms.

https://creativecommons.org/licenses/by/4.0/

The licence covers this repository's own code and the data files shipped in it.
It does not cover the electricity demand and VRE time series you download
separately from ENTSO-E and Copernicus, which keep their providers' terms, nor
the Backbone files in `src_files/GAMS_files/`, which carry their own LGPL-3.0
headers. [ATTRIBUTION.md](ATTRIBUTION.md) records where each data source comes
from and what was changed.


## Citation

Lindroos, T.J., Ikäheimo, J., Purhonen, A., Rämä, M., Hiltunen, P., Harrison, E., Jasiūnas, J., and Kumpulainen, T. North European energy system model. https://github.com/vttresearch/north_european_model

Cite the version you used; the versions are listed below. [CITATION.cff](CITATION.cff)
carries the same details in machine-readable form.


## Version history

The model is cited by version. v1 and v2 predate this repository's git history,
which begins in March 2023.

### v3 -- in development

See [CHANGELOG.md](CHANGELOG.md) for what has changed.

### v2 -- a major rewrite of v1

Documented in:

* Lindroos, T. J., & Ikäheimo, J. (2024). Profitability of demand side management
  systems under growing shares of wind and solar in power systems. *Energy Sources,
  Part B: Economics, Planning, and Policy*, 19(1).
  https://doi.org/10.1080/15567249.2024.2331487

and used in:

* Hiltunen, P., Lindroos, T. J., & Rämä, M. (2025). The impact of electric boilers
  and heat storages in the Nordic power markets and district heating systems.
  *Cleaner Engineering and Technology*, 27, 101028.
  https://doi.org/10.1016/j.clet.2025.101028
* Jasiūnas, J., & Lindroos, T. J. (2026). Powering future Europe through variable
  renewable energy droughts. *Energy Conversion and Management*, 358, 121493.
  https://doi.org/10.1016/j.enconman.2026.121493
* Kiehle, J., Lindroos, T. J., Louis, J.-N., & Pongrácz, E. (2026). Lost flexibility:
  Hydropower capabilities in large-scale, Pan-European Energy system models.
  *Applied Energy*, 423, 128324. https://doi.org/10.1016/j.apenergy.2026.128324
* Harrison, E., Rasku, T., Kiviluoma, J., & Helistö, N. Energy system impacts of
  globally versus locally optimised residential heating and cooling demand response
  in Finland across multiple weather years -- A North European case study. SSRN.
  http://dx.doi.org/10.2139/ssrn.6471140
* Rasku, T., & Louis, J.-N. (2026). Capacity expansion prospects of small modular
  light water reactors for electricity and district heat production in Europe.
  *2026 22nd International Conference on the European Energy Market (EEM)*,
  Trondheim, Norway, 1-6. https://doi.org/10.1109/EEM68581.2026.11589630

### v1 -- the first version

Used in:

* Rasku, T., & Kiviluoma, J. (2019). A Comparison of Widespread Flexible Residential
  Electric Heating and Energy Efficiency in a Future Nordic Power System. *Energies*,
  12(1), 5. https://doi.org/10.3390/en12010005
* Lindroos, T. J., Mäki, E., Koponen, K., Hannula, I., Kiviluoma, J., & Raitila, J.
  (2021). Replacing fossil fuels with bioenergy in district heating -- Comparison of
  technology options. *Energy*, 231, 120799.
  https://doi.org/10.1016/j.energy.2021.120799


## Support

Contact the authors.


## Documentation

Pages in [docs/](docs/). Add a new page here as well as in the folder, or nothing
links to it.

- [Source workbook conventions](docs/source-workbook-conventions.md) — how the builder
  reads the Excel files in `src_files/data_files/`: marking rows and columns as not
  input, where a sheet ends, what happens to a cell that should be a number and is not,
  how `method` combines rows from several files, how a node and a unit get their names,
  and why renaming anything a formula keys on needs checking by number.
- [The source data phase](docs/source-data.md) — what the builder does with those files:
  the order the steps run in and why, how file order and Excel tab order together decide
  which row wins, what excluding a node takes with it, and what the phase reports about a
  column nothing reads.
- [Timeseries](docs/timeseries.md) — how the build turns any hourly data source into
  Backbone input: what a processor is responsible for and what the shared pipeline does,
  what climate years and windows are, why a zero is the hard case, and what is checked
  before anything is written. Start here, then read the page for the source you care
  about.
- [Hydro data](docs/hydro.md) — the four hydro types and what they simplify away, which
  file supplies which number and in what unit, which seasonal limits are not built and
  why, and how gaps in the PECD data are repaired or refused.
- [District heating demand timeseries](docs/dh-demand-timeseries.md) — the whole
  calculation from outdoor temperature, why `TWh/year` is a normal-year figure and no
  single climate year reproduces it, what a zero hour would mean, and which countries
  can be built.
- [Electricity demand timeseries](docs/elec-demand-timeseries.md) — how a TYNDP profile
  becomes each node's hourly demand, why `Constant_share` is blank, which countries and
  climate years exist, and what the parquet cache proves before it is trusted.
- [Wind and solar timeseries](docs/vre-timeseries.md) — what a PECD download decides and
  why the CSV cannot tell you, why one folder may hold only one download, how a node is
  given one of several PECD zones, and what a zero capacity factor means.
- [Input Excel builder](docs/input-excel.md) — the last phase: which sheets `inputData.xlsx`
  gets, why a parameter column is missing whenever nothing set it, how a node is decided to
  be a price or a balance or a storage node, which capacities and storage start levels are
  derived rather than read, and what each of its warnings is asking you to change.
- [Identified gaps](docs/identified-gaps.md) — a working inventory rather than a
  reference page: what Backbone can express that this build does not write, and which
  of its own rules are known to be provisional. Read it before designing anything that
  adds a parameter or a sheet.
- [Running the model](docs/running-the-model.md) — how a built input folder becomes
  a solved model: the one command that starts a run, why runs happen from the Backbone
  checkout above rather than from here, why two runs of one scenario corrupt each other,
  how to tell a run worth trusting from one that merely finished, and what the parent
  checkout's skills and scripts already cover.
- [Migration guide](docs/Migration%20guide.md) — what to change in a workbook or a
  config when an input format changes, newest entry last.

`tools/` holds standalone scripts that answer a question about a build rather than
taking part in one, each documenting itself in its module docstring:

- `compare_source_workbooks.py` — two versions of the source workbooks compared
  numerically, with `--git-ref` to take the earlier one from git. The way to review a
  deliberate data edit, since a binary `.xlsx` has no readable diff, and the only thing
  that catches a renamed value that a `SUMIF` or `VLOOKUP` still keys on.
- `input_data_summary.py` — what is in one built folder, written as a `report.md` with
  its figures into a subfolder of it: capacity, demand, storage, hydro, interconnection
  and prices by country, plus a net-load duration curve and what 35 weather years do to
  the numbers. Its hydro section checks, per bidding-zone store, whether inflow can
  carry the minimum generation and whether a full store can pass its inflow. Run it
  after a build to see the scenario you just produced, or hand the folder to a
  colleague who was not going to run Python. Run it as
  `python build_input_summary.py <built_folder>` from the model folder; that wrapper is
  the same tool under a name that sits beside `build_input_data.py`.
- `check_unittype_columns.py` — a folder of workbooks checked against the unittype rule.
- `compare_input_excels.py` and `compare_workbook_parts.py` — two `inputData.xlsx` files
  compared on values, and as zip archives part by part.
- `profile_build.py` — a build run under a profiler, reported by phase.
- `prepare_zone_geometry.py` — the two map assets `input_data_summary.py` draws on, one
  per level, built once by hand from Natural Earth plus an ENTSO-E bidding-zone layer and
  committed to `tools/maps/`. Where they come from and what was changed is recorded by
  hand in [ATTRIBUTION.md](ATTRIBUTION.md) at the repository root.
  Run it only to change what the maps look like, never as part of a build.

For the model parameters themselves, see `docs/dictionary.md` and `docs/features.md`
in the Backbone repository. For anyone changing the pipeline rather than the data,
`tests/README.md` carries the NA/zero boundary map.


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
	

The installed environment needs one additional package as Miniconda does not automatically find it. After creating and activating the `northEuropeanModel` environment, install the following additional package in Miniconda by typing:

```
pip install gamsapi[transfer]==xx.y.z
```

where xx.y.z is your GAMS version. You can find the correct version by opening GAMS Studio, clicking **Help** -> **GAMS Licensing** -> check GAMS Distribution xx.y.z.

After these steps, you should have the required software and environment ready.

`environment.yml` covers both building the input data and the figures
`tools/input_data_summary.py` draws. If you already have a `northEuropeanModel`
environment from an earlier version, run the update command in the next section to pick
up packages added since.

 
## Updating the conda environment

[Back to top](#North-European-energy-system-model)

Some updates might require updating the conda environment. This is relatively easy process when following these steps:
  * Open the installed Miniconda Prompt (e.g. type `miniconda` or `anaconda` to windows search bar), 
  * In Miniconda, go to folder **backbone/north_european_model/** by typing two commands: `c:` and then `cd c:\backbone\north_european_model`.
  * In Miniconda, update the environment by running following:
	
  ```
	conda env update -n=northEuropeanModel --file=environment.yml
  ```
	
Follow the instructions in the dialogue and install the required updates, if there are any. No further actions are needed.



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


Once it finishes, read what you built by typing
`python build_input_summary.py <output_folder>`, using the output folder named below.
That writes a `summary/report.md` inside it with the figures beside it: how far each
carrier reaches, capacity and demand by country, storage, interconnection, and what the
35 weather years do to the numbers. It only reads the build; everything it writes goes
into that one subfolder, which it overwrites on every run.

At the time of writing, the created "National Trends" takes about 500 Mb, is generated in ~7 minutes, and has ~300 files. Writing some larger sets of GDX files might take up to 60 seconds and the code might seem stuck for those periods, but should eventually proceed.

The `config_NT2030.ini` writes output files to **'backbone\north_european_model\input_National Trends_2030\'** folder. 


You can run Backbone either directly from the created output folder or by copying these files to **backbone\input**, see instructions from [Running Backbone](#running-backbone).


### Choosing VRE processor

Current config files use PECD timeseries through the `VRE_PECD` processor. It is not recommended to edit config files stored in GIT, but instead take a copy, rename it, and edit your own file.

Timeseries processors are selected and configured in the `timeseries_specs = {}` dictionary in config files. The configuration for onshore wind in `config_NT2030.ini` looks like this:

	'wind_onshore': {
		'processor_name': 'VRE_PECD',
		'bb_parameter': 'ts_cf',
		'bb_parameter_dimensions': ['flow', 'node', 'f', 't'],
		'custom_column_value': {'flow': 'onshore'},
		'gdx_name_suffix': 'wind_onshore',
		'rounding_precision': 5,
		'input_sub_folder': 'PECD-onshore/',   # folder, not file
		'attached_grid': 'elec',
	},

The keys each spec accepts are documented in the comment block above `timeseries_specs` in any shipped config file.

**The older `VRE_MAF2019` processor has been removed.** A config still naming it will fail to load that processor and write no timeseries for it. Switch the spec to `VRE_PECD` as above; its `PECD-MAF2019-wide-*.csv` inputs are no longer read by anything.



### Checking run specification files

The script automatically copies the following required run specification files from `src_files\GAMS_files` to `<output_folder>`, and the user is free to edit them afterwards. In most cases, users do not need to edit these at all.
* `1_options.gms` - some solver settings documented inside the file
* `timeAndSamples.inc` - sets defining timestep and forecast domains in Backbone 
* `modelsInit_example.gms` - a default modelsInit file calling scheduleInit.gms
* `scheduleInit.gms` - a tailored scheduleInit file for the Northern European Backbone
* `changes.inc` - reads possible additional excel data, reads timeseries gdx files, and allows users to add their own project specific changes to the end of the file

The python script constructs following files
* `import_timeseries.inc` - this is a specific file containing instructions for Backbone about how to import timeseries GDX files

**Note:** The included `scheduleInit.gms` and `changes.inc` files have a specific structure to make them work with *climateYear* and *modelledDays* parameters. If using your own files, adapt a similar structure to them.

You can double check the contents of these files, but this is not needed if the default settings are ok.




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

### Choosing file location

Run the model by running Backbone.gms in GAMS. You can copy created input files to backbone\input or run the directly from the created <output_folder>.

It is recommended to run the model directly ftom <output_folder> to guarantee the most recent files, allow running different scenarios from different folders, etc.

### Command line parameters 

The model supports the following command line options (use two hyphens in the beginning)
* `--input_file_excel` is a mandatory parameter for defining the used input Excel file name (e.g. inputData.xlsx)
* `--climateYear` [0, 1982-2016]. Default 2015. This parameter allows a quick selection of which time series year the model uses for profiles and annual demands and water inflows. Giving this parameter greatly reduces the solve time as the model drops ts (time series) data from other years and loops the selected time series year. By giving value 0, user can run the model with multiyear time series, but the user is responsible for giving the correct starting time step and checking for error. This feature (tsYear=0) is untested.
* `--modelledDays` [1-365]. Default 365. This option defines the amount of modelled days. If used with tsYear, the maximum value is 365. Otherwise user can give longer time periods, but must check that original timeseries length will not be exceeded.
* `--forecasts` [1, 3]. Default 3. Sets how many forecast branches the model carries beside the realized time series, and requires the 10p, 50p, and 90p time series files in the input folder. Accepted values are 1 (realized values and 1 central forecast) or 3 (realized values, 1 central forecast, 1 difficult forecast, 1 easy forecast). It is recommended to use 3 forecasts due to improved hydro power modelling.
* `--input_dir` allows setting a custom location for the input directory. The default value is 'input' pointing `backbone\input` by default. 
* `--output_dir` allows setting a custom location for the output directory. The default value is 'output' pointing `backbone\output` by default.

See the full list of available command line parameters from Backbone's [documentation](https://gitlab.vtt.fi/backbone/backbone/-/blob/master/docs/running-backbone/command-line-parameters.md).


### Examples

Working command line options for `backbone.gms` would be, for example:
* running the model directly from <output_folder>, full year, climate year 1995: `--input_dir=".\north_european_model\input_ObservedTrends_2030" --input_file_excel=inputData.xlsx --climateYear=1995`
* Running the model from `.\backbone\input` with all default assumptions: `--input_file_excel=inputData.xlsx`
* running the selected climate year, 1 week test: `--input_file_excel=inputData.xlsx --modelledDays=7 --climateYear=1995`


**NOTE:** Use " instead of ' when writing e.g. folder names with spaces. For example, --input_dir='.\dir with spaces' does not work in many workflows, but --input_dir=".\dir with spaces" should work.







## Tool assisted running

[Back to top](#North-European-energy-system-model)

The sections above are what a modeller needs to run the model by hand. This one is for
anyone wanting to automate the runs, drive them from a script or an AI assistant, or
change the pipeline rather than the data.

**Call the interpreter by its full path.** Automation has no Miniconda Prompt to open and
no `conda activate` step, so name the `northEuropeanModel` environment's `python.exe`
directly, or go through `conda run -n northEuropeanModel python ...`. Both the build and
the tests work that way:

```
<path to northEuropeanModel>\python.exe build_input_data.py src_files config_NT2030.ini
<path to northEuropeanModel>\python.exe -m pytest
```

**Starting a run.** `run_model.py` is the one command, and [Running the
model](docs/running-the-model.md) is the page behind it: where a run's output goes, why
two runs of one scenario corrupt each other, how to tell a run worth trusting from one
that merely finished, and what the Backbone checkout above already covers.

```
python run_model.py --list
python run_model.py OT2030 --year 1998 --days 1
```

It runs the model through that checkout, which owns GAMS, the solver and the per-run
isolation; nothing here assembles a `gams` command line of its own. The `run-*.cmd`
files stay the quick by-hand route, **one at a time** — they end with a `cmd` line that
opens an interactive shell, which is convenient by hand and a hang that never ends under
a script.

**The check worth automating.** A one-day run on OT2030 is the cheapest thing that proves
the whole chain end to end — a build, then `python run_model.py OT2030 --year 1998
--days 1`. From cold the two take about ten minutes together; with the build cache warm
the build alone is about two minutes and the solve about twenty seconds. It needs the
[time series downloaded earlier](#downloading-required-time-series-files) to be in place;
nothing in the pipeline can produce those.

**Telling whether the run worked.** A run that finishes is not a run to trust: the exit
code, then `warnings.log` in full, then the solve status, the realised cost and the
dummy tables. The order and the traps are in [Running the
model](docs/running-the-model.md), and the Backbone checkout's `backbone-result-reader`
skill owns the ones that quietly give a wrong number. `tools/input_data_summary.py`
describes a *build*, not a result; it is not the tool for this.

**Running the tests.** See [tests/README.md](tests/README.md). About four minutes warm,
but fifteen or more from cold, and it checks the pipeline's internals rather than a
finished build — so it is what to run when changing `src/`, rather than the routine gate.
Its header prints whether `gams.transfer` is the real API or a stub: on a machine without
GAMS a green run has silently skipped every GAMS test, and that line is the only thing
that says so.

**Checking the environment.** The `northEuropeanModel` environment covers the Python
half. Installing GAMS, choosing a solver and matching `gamsapi[transfer]` to it belong to
the Backbone checkout this model sits inside — its `AGENTS.md` and
`.claude/skills/backbone-quickstart/` cover all three, and none of it is repeated here.
Normally there is nothing to configure: `src/GDX_exchange.py` binds `gams.transfer` to
the GAMS install matching the `gamsapi` you pinned. If GDX reads or writes go wrong, that
skill's `scripts/check_env.py` reports what the machine actually has.

**Write down what works, where it cannot be shared.** Once the build and the tests are
proven, record the interpreter, those commands and whether the downloaded time series are
present in `local-setup.txt` at the model root. It is gitignored, so it describes your
machine and travels to nobody. Head it with the date, and with the note that if it
disagrees with `check_env.py` the probe is right and the file is stale. Nothing
machine-specific belongs in a tracked file — not this README, not `environment.yml`, not
`docs/` and not the tests, because those are shared and would go stale for everyone else.

**Tools.** `tools/` holds standalone scripts that answer a question about a build rather
than taking part in one; they are listed under [Documentation](#documentation). Look
there before writing a throwaway script.
