
Electric vehicles input data
=======

This script creates time series csv files for EV charging.

Requires
	pandas and openpyxl which can be installed with pip.
	PI_calculations.xlsm and region_coefficients.csv as input files.

In region_coefficients.csv, non-existent or 0 coefficients are skipped and no time series will be created for these areas.




