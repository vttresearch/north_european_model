# CLAUDE.md -- North European Energy System Model

## What is this project?

This repository builds input data for the Backbone energy system model, modelling European power systems including district heating, hydrogen, etc. It reads scenario data from Excel files and time series sources, processes them through a Python pipeline, and produces a Backbone-compatible input Excel plus GDX time series files.

## Scope for AI assistance

Only the following directories contain actively developed code and data definitions:
- `src/` -- Python source code (data pipeline, processors, utilities)
- `src_files/` -- configuration files (.ini), Excel data files, GAMS templates, time series

All other subdirectories are generated outputs or ad-hoc analysis folders -- skip them.

All `.cmd` files are user-specific run scripts -- skip them.

## Repository structure

```
build_input_data.py          Main entry point 

src/
  config_reader.py           Reads .ini config files into a dict. Ensures that all config parameters exist and have valid default assumptions.
  build_input_excel.py       Builds the final Backbone input Excel (BuildInputExcel class)
  data_loader.py             Data loading utilities
  GDX_exchange.py            GDX read/write helpers
  json_exchange.py           JSON read/write helpers
  hash_utils.py              File hashing for cache invalidation
  utils.py                   Shared utilities
  pipeline/
    cache_manager.py                Tracks which pipeline steps need re-running
    logger.py                       collects log messages
    source_excel_data_pipeline.py   Reads and merges source Excel files
    bb_excel_context.py             Context object passed to BuildInputExcel
    timeseries_pipeline.py          Orchestrates time series processing
    timeseries_processor.py         Runs individual processor classes
  processors/                individual processors and an abstract class of processors 

src_files/
  config_*.ini               Scenario configs (NT2030, NT2040, OT2030, H2heavy, etc.)
  data_files/                Excel source data (unit types, fuels, demands, transfers, storage, etc.)  -- partly local workfiles unsynchronized to git
  data_scripts/              One-off data processing scripts and raw TYNDP sources
  GAMS_files/                GAMS templates copied to output folders
  timeseries/                Large time series files (CSV, parquet, xlsx) -- partly gitignored
```

## Execution flow

1. User runs python build_input_data.py <input_folder> <config.ini>
2. Config is parsed (`config_reader.py`) defining scenarios, years, country codes, file lists, and time series specs
3. For each (scenario, year, alternative) combination:
   - **Logger** -- `logger` collects log messages from the run and is passed to all pipelines 
   - **Cache check** -- `CacheManager` determines which steps need re-running
   - **Source Excel phase** -- `SourceExcelDataPipeline` reads and merges data Excel files
   - **Time series phase** -- `TimeseriesPipeline` runs each processor defined in `timeseries_specs`
   - **Build Excel phase** -- `BuildInputExcel` assembles the final `inputData.xlsx`
   - **Finalize** -- GAMS template files are copied to the output folder
4. Output goes to `<output_folder_prefix>_<scenario>_<year>[_<alternative>]/`


## Data conventions

The two main pipeline stages use **different** NA/zero conventions. Mixing them up is a common source of bugs.

### SourceExcelDataPipeline (`source_excel_data_pipeline.py`)

**NA and 0 are distinct.**

- `pd.NA` means "no data" — the cell was empty in the source Excel.
- `0` means "explicitly set to zero" — the user wrote 0 in the cell.

This distinction is necessary so that users can intentionally overwrite an inherited or default value with zero (via `method=replace`). A merge that collapses NA and 0 would make that impossible.

After `normalize_dataframe`, numeric columns use pandas `Float64` dtype: missing cells become `pd.NA`, explicit zeros remain `0.0`. String/object columns use `pd.NA` for missing values; empty strings are not used.

### BuildInputExcel (`build_input_excel.py`)

**0 = NA = None = "parameter not set".**

By the time data reaches `BuildInputExcel`, the distinction between an empty cell and an explicit zero is no longer meaningful. Backbone treats an absent parameter and an explicit 0 identically for all parameters whose Backbone default is 0.

Numeric columns are handled in three steps:
1. `_coerce_numeric_dtypes()` casts all known numeric parameter columns in the source DataFrames to `Float64`, coercing non-numeric values to `pd.NA`. It does not fill defaults.
2. Each `create_*()` function applies non-zero defaults from `PARAM_*_DEFAULTS` (e.g. `isActive = 1`) to its own output DataFrame via `fillna`. Applying defaults here — rather than in `_coerce_numeric_dtypes()` — ensures they are enforced for all rows regardless of which data source contributed them (source DataFrames, time series, inferred unit/demand data, etc.).
3. `utils.fill_numeric_na()` is called at the end of each `create_*()` to convert any remaining `pd.NA` to `0` before writing to Excel.

## Error handling policy

The pipeline distinguishes two phases based on whether the logger has been initialized:

- **Before logger initialization** (config reading, argument parsing): raise exceptions and abort. There is no logger to report to and no meaningful partial run to continue.
- **After logger initialization** (all pipeline phases): do not raise. Log a warning via `logger.log_status(message, level="warn")` and continue with a safe default (empty DataFrame, skipped output file, etc.). The goal is to finish the run and deliver a complete log so the user can diagnose all problems at once rather than fixing them one by one.

`ProcessorRunner` in `timeseries_processor.py` enforces this for the timeseries phase: exceptions from processor `__init__`, `run_processor()`, and GDX writing are all caught and logged as warnings.

## Conventions

- Follow local style in each file.

## Don't

- Access folders above the workspace root
- Run Python scripts or any shell commands. Do not execute `python`, `py`, `pip`, or any other CLI tool. Read and edit files only.