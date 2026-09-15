:: Setting the climate year and number of days run
set "Y=1998"
set "D=1"

Title NE Backbone, OT2030, %Y%, %D%d

cd..

:: RUN THESE ONE AT A TIME.
:: Each run gets its own GAMS scratch folder (scrdir), which keeps GAMS's own
:: temporary files apart: the compiled model, the solver input and output, and work
:: gdx files. That alone does not make two runs safe. Every run of a scenario
:: prepares its input in the same <input_dir>\tempFiles\ under fixed names, and all
:: of these scripts write into the same results folder. Two at once overwrite each
:: other and both still report numbers, so the failure is silent rather than an error.
:: output_file and debug_file are named per run so the result files at least survive.
::
:: For sweeps, batches, or anything unattended, use run_model.py instead: it gives
:: every run its own output folder and refuses to start a second run against an input
:: folder already in use. See docs/running-the-model.md.

if not exist ".\north_european_model\scratch\OT2030-%Y%-%D%d" mkdir ".\north_european_model\scratch\OT2030-%Y%-%D%d"

:: running backbone
gams Backbone.gms ^
--input_dir="./north_european_model/input_ObservedTrends_2030" ^
--output_dir="./north_european_model/results" ^
--output_file="results-OT2030-%Y%-%D%d.gdx" ^
--debug_file="debug-OT2030-%Y%-%D%d.gdx" ^
--climateYear=%Y% ^
--modelledDays=%D% ^
--input_file_excel=inputData.xlsx ^
--solver_name=cplex ^
--debug=1 ^
scrdir="./north_european_model/scratch/OT2030-%Y%-%D%d"


cmd
