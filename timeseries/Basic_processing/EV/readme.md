
Electric vehicles input data
=======

# Calculation of time series for EV charging. 

This script creates time series (node, unit and influx time series) csv files for EV charging. 

Requires
	pandas and openpyxl which can be installed with pip.
	PI_calculations.xlsx and region_coefficients.xlsx as input files.

In region_coefficients.xlsx, non-existent or 0 coefficients are skipped and no time series will be created for these areas.

region_coefficients.xlsx lists the number of car groups specified in PI_calculations.xlsx, e.g. 25 cars.

# Usage
Run starthere.py with suitable scenario definition.

# References:
- https://scholar.google.fi/citations?view_op=view_citation&hl=en&user=DWr7TDoAAAAJ&cstart=100&pagesize=100&sortby=pubdate&citation_for_view=DWr7TDoAAAAJ:WF5omc3nYNoC


