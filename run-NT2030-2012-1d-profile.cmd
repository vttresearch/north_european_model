Title NE Backbone, NT2030, 2012, 1d

cd..

:: Each run gets its own GAMS scratch folder (scrdir). GAMS writes all temporary
:: files there: the compiled model, the solver input and output, and work gdx files.
:: With a unique scrdir per run, several Backbone runs can be started in parallel
:: without overwriting each other's temporary files. If the runs shared the default
:: scratch folder, they would fail or return results from the wrong run.
if not exist ".\north_european_model\scratch\NT2030-2012-1d" mkdir ".\north_european_model\scratch\NT2030-2012-1d"

:: running backbone
gams Backbone.gms ^
--input_dir="./north_european_model/input_tyndp2024_NationalTrends_2030" ^
--output_dir="./north_european_model/results" ^
--debug_file="debug-NT2030-2012-1d.gdx" ^
--climateYear=2012 ^
--modelledDays=1 ^
--input_file_excel=inputData.xlsx ^
--solver_name=cplex ^
--debug=1 ^
-profile=4 ^
scrdir="./north_european_model/scratch/NT2030-2012-1d"


cmd
