# Changelog -- North European Energy System Model

Grouped by subject rather than by date.

## Timeseries

Every processor now proves its input data before using it, names what it could
not build and why, and has a documentation page of its own.

- How the build turns an hourly source into Backbone input: [docs/timeseries.md](docs/timeseries.md).
- hydro: built only for the hydro nodes `nodedata` carries; reservoir sizes come
  from `nodedata` `upwardLimit` and `PECD-hydro-capacities.csv` is removed; inflow
  and the seasonal fill limits are continuous over the year change; several
  zero-inflow and zero-limit periods from partial source data fixed; AT00 has its
  seasonal limits back. [docs/hydro.md](docs/hydro.md)
- electricity demand: every scenario year reads the 2030 profiles, because the
  2040 ones contain hours of negative demand. [docs/elec-demand-timeseries.md](docs/elec-demand-timeseries.md)
- district heating: [docs/dh-demand-timeseries.md](docs/dh-demand-timeseries.md)
- wind and solar: [docs/vre-timeseries.md](docs/vre-timeseries.md)
- `VRE_MAF2019` and `hydro_mingen_limits_MAF2019` removed. A config still naming
  either writes no timeseries for it; switch to `VRE_PECD`.
- Demand grids with no processor of their own get a constant `influx` instead of
  a flat timeseries; `ts_influx_other_demands.gdx` is no longer written.
- A processor contributes to the source data tables instead of returning a
  secondary result; the `secondary_output_name` spec field is retired.
  [docs/timeseries.md](docs/timeseries.md)
- A `node`, `grid` or `flow` a processor builds data for that the source data
  does not have is reported.
- wind and solar: capacity factors are built only for the nodes `unitdata`
  attaches a unit of that flow to. [docs/vre-timeseries.md](docs/vre-timeseries.md)
- The build reports what needs acting on rather than what happened: an absence
  the source workbooks already state is silent, and what the rules handled is a
  line of counts. [docs/timeseries.md](docs/timeseries.md)
- `is_input_data_dependent` is retired and a timeseries processor is never copied
  between scenario folders; a config still setting it is ignored, and a
  multi-scenario build costs about a minute more per scenario.
- A processor reruns when the input it is given changed or its own code changed,
  and the build names the value that changed. Editing a cost in `unitdata` no
  longer rebuilds the wind and solar timeseries.
  [docs/timeseries.md](docs/timeseries.md)
- `requires_source_data` names the columns of each table a processor reads, and
  a processor receives only those. The older tuple-of-table-names form still
  works and delivers the whole frame.
  [docs/timeseries.md](docs/timeseries.md)
- `reads_input_files` declares the files a processor opens, so a replaced PECD
  download is noticed. A processor declaring none is rerun every build.
  [docs/timeseries.md](docs/timeseries.md)
- A cold build is about a third faster.

## Source workbooks

- `generator_ID` is removed. A `unitdata` row names its `unittype` directly, and
  `unittypedata` is keyed on `unittype` with a free-text `## Description` column
  in place of the old name.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- `transferdata` is unidirectional: one row per line per direction, with
  `transferCap` in place of `export_capacity` and `import_capacity`. Either old
  column is named by the build and contributes nothing.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- `##` in a cell skips the row, `##` in a column header skips the column.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- A build no longer holds every source workbook open for its whole run, which
  made them read-only in Excel while it ran.
- Excel error values now reach the report at all. pandas turns every error cell
  into an empty one before anything can look, so no `#REF!` or `#N/A` had ever
  been named.
- Malformed cells are reported once per workbook, naming the worst three columns,
  instead of once per column.
- Malformed numbers (`1,000.0`, `100 MW`, `#REF!`) are reported and treated as
  not set, in source excels and timeseries files alike.
- Warnings about a unittype or a node name the workbook and sheet it was written
  in, and say when another sheet spells the same name differently.
- A row with no unittype is identified by its country, scenario and year.
- A blank key column, or a year that is neither a real year nor the `1` meaning
  every year, is an error naming the spreadsheet row. Such a row used to be
  dropped in silence.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- Source workbooks renamed for what they hold: `dheat_balticData.xlsx`,
  `dheat_unitdata_PL_DE_AT.xlsx`, `dheat_unitdata_SE_DK.xlsx`. The unused
  `unitdata_TYNDP-2020.xlsx` and `unitdata_additional-*.xlsx` are removed, and
  some content moved between the remaining ones.
- Unit types and several fuel nodes renamed.
- Finnish city units and district heating transfer links no longer share
  identical costs, and VRE costs are slightly higher.
- Three AT00 units are deactivated with `method = remove` rather than a year that
  matched nothing; the sub-10 MW rule drops one further unit.
- Blank rows, unnamed columns and repeated headers inside a table are reported
  rather than silently dropped.
- A node that only one of `nodedata` and `demanddata` knows about is reported.
- `merge_row_by_row`: column titles compared case-insensitively, first spelling kept.
- Excluding a grid or node says how many units it removed.
- A column no stage reads is reported by file and sheet.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- What the source data phase does with a workbook:
  [docs/source-data.md](docs/source-data.md)
- A demand written as `0` is kept instead of being deleted as an empty row.
- `multiply` leaves a value unchanged when either side is missing, instead of
  writing `0`.
- An `add` or `multiply` row whose key matches nothing is reported; `multiply`
  no longer creates a record.
- A row with no `scenario` or `year` is reported before it is dropped.
- A sheet that overwrites its own earlier row with `replace` is reported.
- A sheet dropped for a missing `country`, `grid` or `unittype` says how
  many rows that cost.
- A user constraint sheet without a `country` column no longer stops the build.
- `emissiondata` is merged on `emission` and `group` together.
- `transferdata` is filtered once for both link ends rather than twice.
- Sheet hashes use the full category prefix, so `unittypedata` is no longer
  hashed as `unitdata` too.
- Config file lists name `Finland_dheat_and_industry.xlsx` with the case the
  folder uses.

## Input excel builder

- How the builder turns the source data tables into `inputData.xlsx`:
  [docs/input-excel.md](docs/input-excel.md).
- p_gn/p_gnn/p_gnu_io/p_unit/p_gnBoundaryPropertiesForStates: empty parameter
  columns are dropped, always keeping one.
- storage starts: a node with no determinable start level is reported. It was
  already left unbounded, but `boundStart=1` and a 0 reference made it look bound.
- Node state boundaries are read from one table, and it states whether each one
  is a constant or a timeseries instead of that following from where it came
  from. [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- The hydro reservoir start level is written by `changes.inc` alone; the input
  excel's provisional value is no longer meant to be the one used.
- `useTimeSeries` and `storageValueUseTimeSeries` spelled `useTimeseries` and
  `storageValueUseTimeseries`, following Backbone.
- `boundStart` is dropped when no node has a storage start level, rather than
  written as a column of zeros.
- Warnings name the first three offenders and then count the rest, instead of
  one line each. [docs/timeseries.md](docs/timeseries.md)

## Test suite

- A pytest suite, and 20+ latent minor bugs found and fixed with it. Current
  scenarios unimpacted.

## Running the model

- `%init_file%` command line parameter, for switching between schedule and invest.
- changes_loop.inc: tighter `vq_userconstraint.up` limits, to speed up the solver.
- `config_OT2030-continuous5y.ini`, an example of `bb_timeseries_length`
  expressions (365*5).
- environment.yml: added matplotlib.
- Toolbox wrapper import fixed by moving the CLI arg parser to `src/utils.py`.
