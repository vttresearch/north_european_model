"""
Transform levels file into required model form.
Change: grid, begintime, endtime, timestep, f, coefficient, inputfilesname and areas according to needs
Output file: s
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime
import calendar

def conv_levels_tomodelform(selected_regions):

    startTime = time.time()
    print('\n')
    print('transforming reservoir level data')

    #Set input values here!
    inputfilename = "input/summary_historical_hydro_reservoir_levels_1h_MWh.csv"
    reservoirfile = "input/region_reservoir.csv"
    grid = "all"
    begintime = "1.1.1982 00:00"  
    timestep = 60             #in minutes
    endtime = "1.1.2018"
    f = ['f00']       #forecast indices
    coef = 1.0   #coefficient with which all the values are multiplied
    param_gnboundarytypes = "reference"

    #select reservoirs, comment the ones that are not needed
    # select specific reservoirs
    region_reservoir = pd.read_csv(reservoirfile)
    region_reservoir = region_reservoir[region_reservoir["region"].isin(selected_regions)]
    selected_reservoirs = region_reservoir["reservoir"].values.tolist()
    print(selected_reservoirs)

    start = pd.to_datetime(begintime)
    end = pd.to_datetime(endtime)

    indf = pd.read_csv(inputfilename, usecols=["Unnamed: 0"])

    for c in selected_reservoirs:
            try:
                cdf = pd.read_csv(inputfilename, usecols=[c])
            except:
                pass
            else:
                indf[c] = cdf[c]

    #copy input files non-leap year and leap-year values appropriately
    indf['time'] =  pd.to_datetime(indf['Unnamed: 0'])
    del indf['Unnamed: 0']

    indf = indf[indf['time']<= end]
    indf['a'] = (indf['time'] - start)/ pd.Timedelta(minutes=timestep)+1


    indf['a'] = indf.a.astype(int)
    dfModel = indf[indf['a']>0].copy() #only positive values of 'a' selected

    # multiply with coefficient
    dfModel.set_index(['time','a'], inplace=True)
    dfModel = coef * dfModel
    dfModel.reset_index(inplace=True)

    #add t, grid, and param_gnboundarytypes
    dfModel['t'] = dfModel['a'].apply(lambda x: f"t{x:06d}") #converted to string
    dfModel['grid'] = grid
    dfModel['param_gnboundarytypes'] = param_gnboundarytypes

    #remove irrelevant columns
    del dfModel['time']
    del dfModel['a']

    result = pd.DataFrame()

    #copy data columns to each f
    dataframes = [dfModel.copy() for i in range(0,len(f))]
    for i in range(0,len(f)):
        dataframes[i]['f'] = f[i]
    result = pd.concat(dataframes)

    result.sort_values(['t'], inplace=True)

    move = result.pop("grid")
    result.insert(0,"grid",move)

    move = result.pop("param_gnboundarytypes")
    result.insert(1, "param_gnboundarytypes", move)

    move = result.pop("f")
    result.insert(2, "f", move )

    move = result.pop("t")
    result.insert(3,"t",move)


    result.reset_index(inplace=True, drop=True)

    #print('\n')
    #print(result.info())
    #print(result.head(10))
        
    result.to_csv('output/bb_ts_historical_levels.csv',index=False)

    print("transformation ", round(time.time() - startTime,2), "s  -- done")
