# CLAUDE.md -- North European Energy System Model

## What is this project?

This repository builds input data for the Backbone energy system model, modelling European power systems including district heating, hydrogen, etc. It reads scenario data from Excel files and time series sources, processes them through a Python pipeline, and produces a folder with all files needed to run the model.


## Scope for AI assistance

Only the following directories contain actively developed code and data definitions:
- `dev/` -- Early versio of developer functions
- `src/` -- Python source code (data pipeline, processors, utilities)
- `src_files/` -- configuration files (.ini), Excel data files, GAMS templates, time series

All other subdirectories are generated outputs or ad-hoc analysis folders -- skip them.

All `.cmd` files are user-owned run scripts. Do not rewrite them unless explicitly asked.


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


## Working inside the Backbone checkout

This project is installed inside a Backbone checkout, so the parent directory `../` is
the Backbone repository. The two are separate git repos.

- **Read freely** from `../` -- `../docs/dictionary.md` and `../docs/features.md` are the
  authoritative parameter/feature references; `../inc/` is the core model logic.
- **Editing anything under `../` needs explicit user confirmation first -- before
  planning the edit, not just before making it.** It is allowed, but it should be rare.
  Default to changing files in this project.
- **Other sessions may work in the Backbone repo concurrently, in folders that change
  from day to day.** Never assume which one is contended; ask before touching it at all.
- **Never write to `../input` or `../output`.** Shared with the Backbone repo and with
  whatever else is running.
- **Never launch GAMS unprompted.** A run from here can collide with a run in another
  stream. See `run-*.cmd`.
