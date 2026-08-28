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

GAMS has no NaN, and a plain `0` **is** empty -- efficient for memory and solve speed, and
correspondingly hard to hold in your head. Python is precise about types; GAMS is not. Nearly
every bug in this project's history lives at that seam, so be explicit about which side of it
you are on.

`0 = NA = None = "not set"` governs **written GDX files too**, not only `inputData.xlsx`.

- **SourceDataPipeline**: `pd.NA` and `0` are distinct. NA = empty cell, 0 = explicitly zero.
  This lets `method=replace` overwrite a value with zero and avoid overwriting with missing data.
- **BBExcelPipeline**: `0 = NA = None = "not set"`. The distinction no longer matters because
  Backbone treats absent and zero identically. `fill_all_na` / `fill_numeric_na` (`src/utils.py`)
  are the crossing point.
- **Timeseries -> GDX**: NaN means "no data" through the whole processor and curing chain. The
  single conversion point is `GDX_exchange.prepare_values_for_gdx`. Do not add a `fillna(0)`
  upstream of it: filling early makes a gap in the source data indistinguishable from a genuine
  zero, and -- because `calculate_climatological_forecasts` takes quantiles, which skip NaN --
  it also biases every forecast branch downward. The conversion is silent during a normal build
  (a source-data gap is not actionable by whoever runs one); `report_missing=True` counts it for
  the timeseries data verifier, whose audience is processor authors.
- **A missing row is not a missing value.** A gap in `value` is legal to the GDX gate; an absent
  *row* is rejected before it. `split_timeseries_to_climate_windows` assigns t-labels by row
  position, so a hole pulls every later hour of that group one label earlier and nothing
  downstream can detect it. `ProcessorRunner` proves per parameter that every group is one
  complete hourly grid covering the same span as the others
  (`timeseries_helpers.find_time_axis_defects`); holes, repeats, sub-hourly rows and ragged
  spans are errors with no config override.

### Dtypes

`utils.standardize_df_dtypes` leaves only `Float64`, `object` and `string`. An **all-NA column is
`object`, never `Float64`**: that means "no assumption has been made", and it is the fix for a
cascade bug where empty text and empty numeric columns became indistinguishable and downstream
code crashed on the dtype it did not expect.

The obligation this creates is on consumers: tolerate an all-NA `object` column where you expect
`Float64`. Never write a `{column: dtype}` map -- state dtype rules as properties.

Object columns use `pd.NA` for missing, never `None` and never `float('nan')`.

`tests/README.md` carries the full boundary map and the assertion rules; the contract is
enforced by `tests/_common/contracts.py` and swept over every loader function.


## Error handling policy

- **Before logger init** (config, arg parsing): raise and abort.
- **After logger init** (pipeline phases): never raise -- log a warning and continue with a safe default.


## What a build says

A warning asks the reader to change something; if there is nothing they can do, it is not
one. **Absence is not a defect** -- the source workbooks state what the model contains, so
a country with no district heating or no offshore wind is silent. What earns a line is
*partial* data: the model has the node or unit and the data for it is missing or
contradictory -- and such a warning **names the first three offenders, then counts
the rest** (`utils.summarise`), because "1 node has no price data" only makes its
reader ask which one. Everything expected and handled costs counts, not names, and
never reasons -- one short line per processor, with the names and the reasoning in
the documentation page. A check that fires on correct data every run is not strict, it is broken.

The full rule, with examples, is "What a build says" in `docs/timeseries.md`.


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
