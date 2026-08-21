# Changelog -- North European Energy System Model

## 2026-08-21
- source data: `##` marks something as the author's, not the model's -- in a data row it skips the row, in a column header it skips the column. Use it for helper tables beside the real one; they are dropped before any validation sees them. **Breaking:** the row marker was a single `#`, which collided with every Excel error value, so a broken formula deleted its own row.
- source data: a cell that should hold a number but does not (`1,000.0`, `100 MW`, `#REF!`) is reported and treated as not set, instead of silently demoting its whole column and dropping rows.
- timeseries processors: `BaseProcessor.read_input_csv`/`read_input_excel` reject an input file with malformed numbers or an inconsistent field count. An unquoted thousands separator no longer shifts every column unnoticed.
- emissiondata and userconstraintdata: their value columns are now coerced to numbers, as are nodedata's `emission_*` columns.

## 2026-08-10
- p_userconstraint: unused dimension slots are autofilled with '-'. 
- merge_row_by_row: column titles are now compared case-insensitively, keeping the first spelling. 
- merge_row_by_row: 'add-non-negative' does no longer zero previous negative values if there is a column header, but no number in it.
- p_gn/p_gnn/p_gnu_io/p_unit/p_gnBoundaryPropertiesForStates: empty parameter columns are dropped, always keeping minimum one column. 
- unitdata: a generator_id missing from unittypedata is now reported instead of silently losing its defaults.
- storage starts: a node with no determinable start level is now reported. It was already left unbounded, but boundStart=1 and a 0 reference made it look bound.
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


