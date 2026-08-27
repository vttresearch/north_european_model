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

## Source workbooks

- `##` in a cell skips the row, `##` in a column header skips the column.
  [docs/source-workbook-conventions.md](docs/source-workbook-conventions.md)
- Malformed numbers (`1,000.0`, `100 MW`, `#REF!`) are reported and treated as
  not set, in source excels and timeseries files alike.
- Blank rows, unnamed columns and repeated headers inside a table are reported
  rather than silently dropped.
- A node that only one of `nodedata` and `demanddata` knows about is reported.

## Input excel builder

- merge_row_by_row: column titles compared case-insensitively, first spelling kept.
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
