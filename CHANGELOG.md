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
