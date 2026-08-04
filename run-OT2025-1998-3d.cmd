Title NE Backbone, OT2025, 1998, 3d

cd..

:: Each run gets its own GAMS scratch folder (scrdir). GAMS writes all temporary
:: files there: the compiled model, the solver input and output, and work gdx files.
:: With a unique scrdir per run, several Backbone runs can be started in parallel
:: without overwriting each other's temporary files. If the runs shared the default
:: scratch folder, they would fail or return results from the wrong run.
if not exist ".\north_european_model\scratch\OT2025-1998-3d" mkdir ".\north_european_model\scratch\OT2025-1998-3d"

:: running backbone
gams Backbone.gms ^
--input_dir="./north_european_model/input_ObservedTrends_2025" ^
--output_dir="./north_european_model/results" ^
--debug_file="debug-OT2025-1998-3d.gdx" ^
--climateYear=1998 ^
--modelledDays=3 ^
--input_file_excel=inputData.xlsx ^
--solver_name=cplex ^
--debug=1 ^
-profile=4 ^
scrdir="./north_european_model/scratch/OT2025-1998-3d"


cmd
