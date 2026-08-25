# Changelog -- North European Energy System Model

## 2026-08-25
- timeseries: how the build turns any hourly data source into Backbone input is now documented, see [docs/timeseries.md](docs/timeseries.md).
- timeseries: build log messages name what was affected and what to do about it; nodes that could not be built are grouped by reason.

## 2026-08-24
- electricity demand: every scenario year now reads `elec_2030_National_Trends.xlsx`; the 2040 profiles are not used, because they contain hours of negative demand. Improved checks and warnings, documentation added, see [docs/elec-demand-timeseries.md](docs/elec-demand-timeseries.md).
- VRE: Improved checks and warnings, documentation added, see [docs/vre-timeseries.md](docs/vre-timeseries.md).

## 2026-08-23
- hydro: timeseries are built only for hydro nodes that `nodedata` carries. See [docs/hydro.md](docs/hydro.md).
- hydro: inflow and the seasonal fill limits are now continuous over the year change.

## 2026-08-22
- district heating: Improved checks and warnings, documentation added, see [docs/dh-demand-timeseries.md](docs/dh-demand-timeseries.md).

## 2026-08-21
- hydro: reservoir sizes come from source excel `nodedata` `upwardLimit`, `PECD-hydro-capacities.csv` removed. See [docs/hydro.md](docs/hydro.md).
- hydro: fixing several zero-inflow and zero-limit periods arising from partial source data. 
- hydro: AT00 regained its seasonal limits after a fix in timeseries processor.
- source data: `##` in a row skips the row, `##`in a column header skips the column. 
- source data: a wide range of malformed numbers (`1,000.0`, `100 MW`, `#REF!`, etc) checked and reported clearly both for source excels and timeseries data.
- source data: blank rows and unnamed columns inside a table are reported, not silently dropped.

## 2026-08-10
- merge_row_by_row: column titles are now compared case-insensitively, keeping the first spelling. 
- p_gn/p_gnn/p_gnu_io/p_unit/p_gnBoundaryPropertiesForStates: empty parameter columns are dropped, always keeping minimum one column. 
- storage starts: a node with no determinable start level is now reported. It was already left unbounded, but boundStart=1 and a 0 reference made it look bound.

## 2026-08-06
- Adding a test suite and fixing 20+ latent minor bugs. The current scenarios are unimpacted.

## 2026-08-04
- environment.yml: added matplotlib
- Added config_OT2030-continuous5y.ini, an example of bb_timeseries_length expressions (365*5).

## 2026-06-24
- Adding %init_file% command line parameters for more flexible switching between schedule and invest
- Updating changes_loop.inc to add more precise limits on vq_userconstraint.up to speed up the solver
- Fixing toolbox wrapper import by moving CLI arg parser to src/utils.py as shared parse_sys_args (was _parse_sys_args in build_input_data.py); .


