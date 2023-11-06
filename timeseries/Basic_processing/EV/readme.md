
Electric vehicles input data
=======

This script creates time series csv files for EV charging.

Requires
	pandas and openpyxl which can be installed with pip.
	PI_calculations.xlsx and region_coefficients.xlsx as input files.

In region_coefficients.xlsx, non-existent or 0 coefficients are skipped and no time series will be created for these areas.

region_coefficients.xlsx lists the number of car groups specified in PI_calculations.xlsx, e.g. 25 cars.


