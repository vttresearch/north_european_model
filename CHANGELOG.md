# Changelog -- North European Energy System Model

## 2026-08-05
- source_data_loader.py: build_from_to_columns no longer raises TypeError when a country or grid cell is not a string (now stringifies, matching build_node_column).
- source_data_loader.py: build_unit_grid_and_node_columns now writes pd.NA, not np.nan, into the object-dtype grid_/node_ columns it creates for unmatched generator_ids.
- utils.py: is_col_empty returns a plain bool instead of numpy.bool_.
- Added dtype/NA contract sweeps over the source-data loaders, plus unit tests for utils, config_reader, logger, hash_utils and _patch_gams_file_content.
- Added tests/README.md: the NA/zero boundary map, assertion rules, and when pinning a value is correct.
- Added pytest suite: pytest.ini, tests/ package, conftest with a gams.transfer stub so most tests run without GAMS, shared FakeLogger and make_config. environment.yml: added pytest.

## 2026-08-04
- environment.yml: added matplotlib, required by analyze_ts.py.
- Added config_OT2030-continuous5y.ini, an example of bb_timeseries_length expressions (365*5).

## 2026-06-24
- Adding %init_file% command line parameters for more flexible switching between schedule and invest
- Updating changes_loop.inc to add more precise limits on vq_userconstraint.up to speed up the solver
- Moved CLI arg parser to src/utils.py as shared parse_sys_args (was _parse_sys_args in build_input_data.py); fixes toolbox wrapper import.

## 2026-06-03
- analyze_ts.py: added district heating (ts_influx_dheat) analysis; dheat subregions ({region}_dheat_{sub}) treated as individual regions, aggregated to national plots.

## 2026-06-02
- analyze_ts.py: added cross-tech annual overview (2x2 boxplot, anomaly, correlation) by country; converts wind/PV CF to MWh via p_gnu_io capacity.

