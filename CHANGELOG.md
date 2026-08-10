# Changelog -- North European Energy System Model

## 2026-08-10
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


