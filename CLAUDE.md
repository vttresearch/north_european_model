# CLAUDE.md -- North European Energy System Model

## What is this project?

This repository builds input data for the Backbone energy system model, modelling European power systems including district heating, hydrogen, etc. It reads scenario data from Excel files and time series sources, processes them through a Python pipeline, and produces a folder with all files needed to run the model.


## Scope for AI assistance

Only the following directories contain actively developed code and data definitions:
- `dev/` -- Early versio of developer functions
- `src/` -- Python source code (data pipeline, processors, utilities)
- `src_files/` -- configuration files (.ini), Excel data files, GAMS templates, time series

All other subdirectories are generated outputs or ad-hoc analysis folders -- skip them.

All `.cmd` files are user-specific run scripts -- skip them.


## Execution flow

1. `python build_input_data.py <input_folder> <config.ini>`
2. Config is parsed defining general settings, input files, and run instructions.
   - Git config files are stored in `src_files/config_*.ini`
3. For each (scenario, year, alternative) combination:
   - **Logger** -- `logger` collects log messages from the run and is passed to all pipelines 
   - **Cache check** -- `CacheManager` determines which steps need re-running
   - **Source data phase** -- `SourceExcelDataPipeline` reads and merges data Excel files
   - **Time series phase** -- `TimeseriesPipeline` runs each processor defined in `timeseries_specs`
   - **Build Excel phase** -- `BBExcelPipeline` assembles the final `inputData.xlsx`
   - **Finalize** -- GAMS template files are edited and copied to the output folder


## Data conventions

The two pipeline stages use **different** NA/zero conventions. Mixing them up is a common source of bugs.
- **SourceExcelDataPipeline**: `pd.NA` and `0` are distinct. NA = empty cell, 0 = explicitly zero. This lets `method=replace` overwrite a value with zero and avoid overwriting with missing data.
- **BBExcelPipeline**: `0 = NA = None = "not set"`. The distinction no longer matters because Backbone treats absent and zero identically.


## Error handling policy

- **Before logger init** (config, arg parsing): raise and abort.
- **After logger init** (pipeline phases): never raise -- log a warning and continue with a safe default.


## Don't

- Access folders above the workspace root.
