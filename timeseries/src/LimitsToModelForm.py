"""
Transform limits file into required model form.
Change: grid, begintime, endtime, timestep, f, coefficient, inputfilesname and areas according to needs
Output file: bb_ts_node.csv
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime
import calendar

def conv_limits_tomodelform(selected_regions):

    startTime = time.time()
    print('\n')
    print('transforming reservoir limits data')

    #Set input values here!
    inputfilename = "input/summary_hydro_reservoir_limits_2015_2016_1h_MWh.csv"
    reservoirfile = "input/region_reservoir.csv"
    grid = "all"
    begintime = "1.1.1982 00:00"  
    timestep = 60             #in minutes
    endtime = "1.1.2018"
    f = ['f00']       #forecast indices
    coef = 1.0   #coefficient with which all the values are multiplied

    #select reservoirs, comment the ones that are not needed
    # notice that DE00_reservoir and DE00_psOpen values are missing and thus produce [0,Inf]
    # not including them would make the limits [0,0] in GAMS
    # select specific reservoirs
    region_reservoir = pd.read_csv(reservoirfile)
    region_reservoir = region_reservoir[region_reservoir["region"].isin(selected_regions)]
    selected_reservoirs = region_reservoir["reservoir"].values.tolist()
    print(selected_reservoirs)


    minval = 0   #when no values in downward limit
    maxval = 0 #when no values in upward limit, in some cases could consider 'Inf'
    minvariable = 'downwardLimit'
    maxvariable = 'upwardLimit'

    start = pd.to_datetime(begintime)
    end = pd.to_datetime(endtime)

    indf = pd.read_csv(inputfilename, usecols=["Unnamed: 0", "boundarytype"])

    for c in selected_reservoirs:
            try:
                df = pd.read_csv(inputfilename, usecols=[c])
            except:
                indf[c] = np.nan
            else:
                indf[c] = df[c]

    #copy input files non-leap year and leap-year values appropriately
    indf['Unnamed: 0'] =  pd.to_datetime(indf['Unnamed: 0'])
    all_res = []
    for y in range(start.year, (end.year+1)):
        if calendar.isleap(y):
                addf = indf[indf['Unnamed: 0'].dt.year == 2016].copy()
                
        else:
                addf = indf[indf['Unnamed: 0'].dt.year == 2015].copy()
        addf['time'] = addf['Unnamed: 0'].apply(lambda x: x.replace(year = y))
        all_res.append(addf)     
    df = pd.concat(all_res)
    del df['Unnamed: 0']

    df = df[df['time']<= end]
    df['a'] = (df['time'] - start)/ pd.Timedelta(minutes=timestep)+1

    #dfModel = pd.DataFrame()
    result = pd.DataFrame()

    df['a'] = df.a.astype(int)
    dfModel = df[df['a']>0].copy() #only positive values of 'a' selected

    # multiply with coefficient
    dfModel.set_index(['time','boundarytype','a'], inplace=True)
    dfModel = coef * dfModel
    dfModel.reset_index(inplace=True)

    #add t and grid
    dfModel['t'] = dfModel['a'].apply(lambda x: f"t{x:06d}") #converted to string
    dfModel['grid'] = grid

    #remove irrelevant columns
    del dfModel['time']
    del dfModel['a']

    #replace nans
    df1 = dfModel[dfModel['boundarytype'] == minvariable].fillna(minval)
    df2 = dfModel[dfModel['boundarytype'] == maxvariable].fillna(maxval)
    dfcomb = pd.concat([df1,df2])

    #copy data columns to each f
    dataframes = [dfcomb.copy() for i in range(0,len(f))]
    for i in range(0,len(f)):
        dataframes[i]['f'] = f[i]
    result = pd.concat(dataframes)
    result.rename(columns={"boundarytype":"param_gnboundarytypes"}, inplace=True)

    result.sort_values(['t','param_gnboundarytypes'], inplace=True)

    move = result.pop("grid")
    result.insert(0,"grid",move)

    move = result.pop("param_gnboundarytypes")
    result.insert(1, "param_gnboundarytypes", move )

    move = result.pop("f")
    result.insert(2, "f", move )

    move = result.pop("t")
    result.insert(3,"t",move)


    result.reset_index(inplace=True, drop=True)

    #print('\n')
    #print(result.info())
    #print(result.head(10))
    #print(result.tail(10))
        
    result.to_csv('output/bb_ts_node.csv',index=False)
    #result.to_excel('summary_reservoir_levels_model_form_MWh.xlsx', index=False)

    print("transformation ", round(time.time() - startTime,2), "s  -- done")
