# CLAUDE.md -- North European Energy System Model

## What is this project?

This repository builds input data for the Backbone energy system model, modelling European power systems including district heating, hydrogen, etc. It reads scenario data from Excel files and time series sources, processes them through a Python pipeline, and produces a folder with all files needed to run the model.


## Scope for AI assistance

Only the following directories contain actively developed code and data definitions:
- `dev/` -- Early version of developer functions, and local reference artifacts
  (a known-good `inputData.xlsx`, reference GDX files). Untracked, and large.
- `src/` -- Python source code (data pipeline, processors, utilities)
- `src_files/` -- configuration files (.ini), Excel data files, GAMS templates, time series
- `tools/` -- shared standalone tools, tracked (see below)

All other subdirectories are generated outputs or ad-hoc analysis folders -- skip them.

All `.cmd` files are user-owned run scripts. Do not rewrite them unless explicitly asked.


## Tools

`tools/` holds standalone scripts that answer a question about a build rather than
taking part in one. **Look here before writing a throwaway script** -- that is what
the folder is for, and a tool that already exists has already been debugged.

Each tool documents itself in its module docstring: what it checks, what it cannot
see, and its usage line. They take command-line arguments, print a report, and exit
0 / 1 so they can gate a loop; they are not imported by `src/` and, with one
exception noted below, are not in the pytest suite. Reference artifacts they compare
against live in the untracked `dev/`, so a usage example may name a file the reader
does not have.

- `input_data_summary.py` -- one built output folder described as a `report.md` with
  embedded figures, written into a subfolder of it. Country-level capacity, demand,
  storage, interconnection and prices from `inputData.xlsx`, plus a net-load duration
  curve and the interannual spread from the per-year GDX files. The only tool that
  reads a build's GDX, and the only one with tests (`tests/unit/test_input_data_summary.py`,
  which covers five arithmetic conventions whose failures a reader could not see).
  It is also the only plotting code in the repo, and the only tool with an entry point
  of its own at the repository root -- `build_input_summary.py`, a thin wrapper that
  adds `tools/` to the path and delegates. The implementation and its documentation
  stay here; keep them here.
- `profile_build.py` -- a build run under a profiler and reported by phase, with the
  caveat that pstats cannot be trusted with it.
- `prepare_zone_geometry.py` -- the two map assets `input_data_summary.py` draws,
  `tools/maps/country_shapes.geojson` and `zone_shapes.geojson`, built by hand and
  committed. Never run by a build. Every border comes from Natural Earth; the
  ENTSO-E layer only says which zone a piece of land belongs to, its own Norway
  outline being 31% sea. Its sources live in the untracked `example_maps/`, so the
  committed assets are the only copy anyone else has.
- `compare_input_excels.py` -- two `inputData.xlsx` files compared sheet by sheet on
  *values*, read as text. Row order does not matter. Blind to formatting.
- `compare_workbook_parts.py` -- two `.xlsx` files compared as zip archives, part by
  part, ignoring only the build timestamps openpyxl stamps into `docProps/core.xml`.
  Sees cell values, column order, widths, alignment, table styles -- use it to prove a
  refactor changed nothing. Literal byte equality is not achievable; those timestamps
  differ on every build.
- `compare_source_workbooks.py` -- two versions of the source workbooks compared
  *numerically*, row by row on their dimension columns, with `--git-ref` to take the
  earlier version straight from git. The one that catches a name still being used as
  a lookup key: `SUMIF`, `VLOOKUP` and `COUNTIF` criteria do not fail when the data is
  renamed, they return 0 or a neighbouring row. Also the way to review a deliberate
  data edit, since a binary `.xlsx` has no readable diff.
- `check_unittype_columns.py` -- a folder of source workbooks checked against the
  unittype rule: no `Generator_ID` header left, every `unitdata` unittype declared by
  some `unittypedata` sheet, and -- with `--legacy-names` -- no cell anywhere still
  holding a pre-migration name, which is how a VLOOKUP helper table gets left behind.
  Reads workbooks rather than a config, so it cannot know which of them a build lists.


## Execution flow

1. `python build_input_data.py <input_folder> <config.ini>`
2. Config is parsed defining general settings, input files, and run instructions.
   - Git config files are stored in `src_files/config_*.ini`
3. For each (scenario, year, alternative) combination:
   - **Logger** -- `logger` collects log messages from the run and is passed to all pipelines 
   - **Cache check** -- `CacheManager` determines which steps need re-running
   - **Source data phase** -- `SourceDataPipeline` reads and merges data Excel files
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


## Environment and how to run things here

**Never guess an interpreter or a command.** Two places hold what has actually been
proven on this machine, and they are the only two:

- `local-setup.txt` at this repository root -- gitignored, so it describes this checkout
  on this machine and travels to nobody. Any assistant can read it, and so can the user.
  Read it first.
- the `python-env` and `test-command-shape` memory blocks, which carry the same facts
  plus the permission-rule shape that lets a command run here without a prompt.

`local-setup.txt` is the fuller of the two: the interpreter as a full path to the
environment's `python.exe`, called directly with no activation step, and the commands
actually proven here, each dated with what it returned and marked when it was not run.

**GAMS needs no configuration here, and this project does not reason about it.** Two
independent things: `gams.transfer` reads and writes GDX from the Python side, and
`GDX_exchange.resolve_gams_system_directory` binds it to the install matching the
installed `gamsapi`, with no environment variable set; separately, a model run takes
whichever `gams` is on PATH, with the solver the `run-*.cmd` files name. *Which* install
and *which* solver those resolve to belongs in `local-setup.txt`, not here. No solver
selection, no fallback and no licence-ceiling arithmetic belongs in this project: the
parent checkout owns that compatibility question, and a second copy of it here would go
stale without anyone noticing.

**If both files are absent, or something GAMS-shaped is actually broken**, the parent's
`backbone-quickstart` skill (`../.claude/skills/backbone-quickstart/`) owns GAMS
detection and the `gamsapi[transfer]` pin, and its `scripts/check_env.py` is the probe.
When the probe and `local-setup.txt` disagree, the probe is right and the file is stale.
Reach for it when something is wrong, not as a routine step -- nothing here needs
configuring while it works.

**Machine specifics belong in those two places and nowhere else.** Never an interpreter
path, environment name, GAMS version or solver choice in `README.md`, `environment.yml`,
`docs/`, `tests/` or this file: those are git-shared and would go stale for everyone
else. That is why this section names none of them.

**Running Python here.** Use `python -m pytest`, not bare `pytest`, so the repository
root lands on `sys.path`; `tests/README.md` has the tiers. The run header prints
`gams.transfer: real API` or `STUBBED`, and that line is the only thing that reveals a
green run which skipped every GAMS test. A bare `python` on PATH is a base interpreter
without pandas -- never install into it. The allow-rules in `.claude/settings.json` are
prefix matches on one interpreter spelling, so a command led with `cd`, `$env:` or a
variable assignment prompts where the same command otherwise would not.

**The end-to-end run needs source data this repository does not ship.** The electricity
demand profiles and the PECD wind and solar downloads are fetched by hand into
`src_files/timeseries/` -- README's "Downloading required time series files" says how.
**Whether a build is runnable at all is therefore a fact about the machine, and
`local-setup.txt` answers it first**; the parent checkout arrives complete, so its own
file never has to. When those downloads are missing the build cannot produce them, the
pytest suite is the only check available, and the answer is to say so rather than to
synthesise inputs or narrow the config until something passes.

**Ask before anything heavy, and know what each one costs.** Warm, the test suite is
about four minutes and an input-data build about two; from cold the suite is fifteen or
more, and a build plus a one-day OT2030 solve about ten. That last is the only check that
proves the whole chain rather than the pipeline's internals, so it is the one worth
automating -- and none of them is free. A GAMS run can also collide with another stream:
see the next section.


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
