# Changelog -- North European Energy System Model

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

