# Changelog -- North European Energy System Model

## 2026-08-10
- p_userconstraint: unused dimension slots are autofilled with '-'. 
- merge_row_by_row: column titles are now compared case-insensitively, keeping the first spelling. 
- merge_row_by_row: 'add-non-negative' does no longer zero previous negative values if there is a column header, but no number in it.
- p_gn/p_gnn/p_gnu_io: empty parameter columns are dropped, always keeping minimum one column. 
- unitdata: a generator_id missing from unittypedata is now reported instead of silently losing its defaults.
- updating .gitignore rules for cleaner definition of the repository

## 2026-08-06
- Adding a test suite and fixing 20+ latent minor bugs. The current scenarios are unimpacted.

## 2026-08-04
- environment.yml: added matplotlib
- Added config_OT2030-continuous5y.ini, an example of bb_timeseries_length expressions (365*5).

## 2026-06-24
- Adding %init_file% command line parameters for more flexible switching between schedule and invest
- Updating changes_loop.inc to add more precise limits on vq_userconstraint.up to speed up the solver
- Fixing toolbox wrapper import by moving CLI arg parser to src/utils.py as shared parse_sys_args (was _parse_sys_args in build_input_data.py); .


