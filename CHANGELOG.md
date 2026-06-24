# Changelog -- North European Energy System Model

## 2026-06-24
- Moved CLI arg parser to src/utils.py as shared parse_sys_args (was _parse_sys_args in build_input_data.py); fixes toolbox wrapper import.

## 2026-06-03
- analyze_ts.py: added district heating (ts_influx_dheat) analysis; dheat subregions ({region}_dheat_{sub}) treated as individual regions, aggregated to national plots.

## 2026-06-02
- analyze_ts.py: added cross-tech annual overview (2x2 boxplot, anomaly, correlation) by country; converts wind/PV CF to MWh via p_gnu_io capacity.

