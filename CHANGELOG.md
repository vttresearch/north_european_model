# Changelog -- North European Energy System Model

## 2026-08-05
- Tests: assert_workbook_consistent now compares keys case-insensitively, as GAMS does, and checks the nodeBalance/usePrice mutual exclusion. Two labels differing only in case are one label to GAMS and a duplicate record that aborts the GDX write.
- source_data_loader: merge_row_by_row no longer degrades every method to a full replace when a category has no numeric column anywhere. An empty 'add' row, and even a 'replace-partial' row, silently blanked the record it was meant to leave alone, and the same row behaved differently depending on whether an unrelated column happened to be numeric.
- source_data_loader: merge_row_by_row now returns the key columns it reports creating. It added them to each input frame but rebuilt the result from a column list computed beforehand, so they were dropped again; create_p_userconstraint believed the message and crashed.
- bb_excel_pipeline: create_p_userconstraint no longer raises KeyError when a userconstraintdata sheet omits unused dimension columns. It detected the missing columns and logged a warning, then selected them anyway and killed the build. A constraint using only the 1st and 2nd dimension is ordinary.
- Tests: added the route tier -- source workbooks to inputData.xlsx, driven in-process with no GAMS and no working-directory dependence. Test workbooks are stored as text (.wb.txt) and built into .xlsx at run time, so fixtures are diffable and cannot rot unnoticed.
- Tests: added workbook delta assertions. Both sides of a comparison are produced by the code under test in the same run, so adding a parameter column or changing a default does not require a test edit. There are deliberately no golden files.
- cache_manager: added timeseries_helpers.py to the watched timeseries files; editing the climate-window or forecast maths did not invalidate the cache, so the build served stale GDX without saying so. Also added the source_data/timeseries inputs and results dataclasses, which were watched for bb_excel but not the other two stages.
- build_input_data: stdout/stderr are set to UTF-8 at entry. Redirecting output to a file (`> build.log`) previously crashed with UnicodeEncodeError on the first log line, because Windows falls back to the locale encoding when the stream is not a console and the status prefixes are non-ASCII.
- GDX_exchange: gams.transfer now binds to the GAMS install matching the installed gamsapi (override with BB_GAMS_API, as in the parent Backbone repo) instead of whichever GAMS was registered last. Removes the "GAMS version differs from the API version" warning on every container. Use new_container(), not gt.Container().
- Timeseries NaN handling: missing values are now converted to 0 in one place only, the GDX boundary (GDX_exchange.prepare_values_for_gdx). Removed the three silent fills that preceded it (ProcessorRunner after validation, cutoff_below as a side effect, calculate_climatological_forecasts after its left join). The conversion is not reported during a normal build: a gap in a source timeseries is not actionable by the person running one. prepare_values_for_gdx(report_missing=True) counts it for the planned timeseries data verifier, aimed at processor authors.
- Behaviour change: where source timeseries have gaps, climatological forecasts differ. Quantiles now skip missing hours instead of counting them as genuine zeros, which previously biased every forecast branch downward.
- GDX writing rejects and logs rows with a blank dimension value (would have become the GAMS set element '') or a non-finite value.
- ProcessorRunner rejects processor output with a missing dimension value, a non-numeric value column, or a time column that cannot be read as datetime. A convertible non-datetime time column is still accepted but now warns.
- ProcessorRunner: an empty processor DataFrame now stops as its message always claimed; previously execution continued.
- calculate_climatological_forecasts no longer adds an 'hour_of_year' column to the caller's DataFrame.
- Removed src_files/config_unittest.ini and src_files/data_files/unitTest.xlsx; superseded by tests/.
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

