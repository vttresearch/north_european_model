"""
copies intermediate csv files from basic processing output folders to timeseries/input
"""


import shutil
import os

# define the files to be copied
srcList = [	
			"./DH/output/DH_2025_timeseries_summary.csv",
			"./DH/output/DH_2030_timeseries_summary.csv",
			"./DH/output/DH_2025_timeseries_average.csv",
			"./DH/output/DH_2030_timeseries_average.csv",
			"./Elec_demand/output/summary_load_2011-2020-1h.csv",
			"./Elec_demand/output/average_load_2011-2020-1h.csv",
			"./Hydro/Inflow/output/summary_hydro_inflow_1982-2020_1h_MWh.csv",
			"./Hydro/Inflow/output/summary_hydro_average_year_1982-2020_1h_MWh.csv",
			"./Hydro/Reservoir_levels/output/summary_historical_hydro_reservoir_levels_1h_MWh.csv",
			"./Hydro/Reservoir_limits/output/summary_hydro_reservoir_limits_2015_2016_1h_MWh.csv",
			"./Hydro/Reservoir_minmax_generation/output/summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv",
    	  ]

# define where each file is copied
dstList = [	
			"../input/DH_2025_timeseries_summary.csv",
			"../input/DH_2030_timeseries_summary.csv",
			"../input/DH_2025_timeseries_average.csv",
			"../input/DH_2030_timeseries_average.csv",
			"../input/summary_load_2011-2020-1h.csv",
			"../input/average_load_2011-2020-1h.csv",
			"../input/summary_hydro_inflow_1982-2020_1h_MWh.csv",
			"../input/summary_hydro_average_year_1982-2020_1h_MWh.csv",
			"../input/summary_historical_hydro_reservoir_levels_1h_MWh.csv",
			"../input/summary_hydro_reservoir_limits_2015_2016_1h_MWh.csv",
			"../input/summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv",
		  ]



# do the copying
for i in range(0,len(srcList)):
	
	src = os.path.normpath(srcList[i])
	dst = os.path.normpath(dstList[i])

	try:
		shutil.copyfile(src, dst)
		print("copied from " + src + " to " + dst)
	except:
		print("\n")
		print("\n")
		print("file not found!!!     : " + src)
		print("\n")
		print("\n")

