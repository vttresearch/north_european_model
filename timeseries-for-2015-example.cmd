Title timeseries for 2015

:: removing old results to avoid a case where model crashes and everything seems to work when old results are copied.
:: NOTE: does not work if these files are open in GAMS
cd..
del input/ts_influx.gdx
del input/ts_cf.gdx
del input/ts_node.gdx


:: running timeseries preprocessing
gams north_europan_model/preprocessTimeseries.gms --input_dir=./input --input_file_excel=bb_input1-3x.xlsx --tsYear=2015


cmd