"""
Transform inflow file into required model form. Change: grid, timeorigin, timestep, f and coefficient according to needs,
Input file: summary_hydro_inflow_1982-2020_1h_MWh.csv
Output file: output/bb_ts_influx_hydro.csv
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import calendar

def conv_inflow_tomodelform(selected_regions):
    
    startTime = time.time()
    print('\n')
    print('transforming hydro inflow data')

    #Set input values here!
    inputfile = "input/summary_hydro_inflow_1982-2020_1h_MWh.csv"
    reservoirfile = "input/region_reservoir.csv"
    grid = "all"
    timeorigin = "1.1.1982 00:00"  
    timestep = 60             #in minutes
    f = ['f00']
    coef = 1.0   #coefficient with which all the values are multiplied

    # select regions for which you want inflow values
    #selected_regions = ['FI00', 'SE01', 'SE02', 'SE03', 'SE04', 'NOS0', 'NOM1', 'NON1', 'DE00']

    indf = pd.read_csv(inputfile)
    indf['Unnamed: 0'] =  pd.to_datetime(indf['Unnamed: 0'])
    indf['a'] = (indf['Unnamed: 0'] - pd.to_datetime(timeorigin))/ pd.Timedelta(minutes=timestep) + 1


    # select specific regions
    region_reservoir = pd.read_csv(reservoirfile)
    region_reservoir = region_reservoir[region_reservoir["region"].isin(selected_regions)]
    selected_reservoirs = region_reservoir["reservoir"].values.tolist()
    print(selected_reservoirs)

    # add missing columns as zero to the input data
    for r in selected_reservoirs:
        if not r in indf.columns:
            print("reservoir ",r, " not found!")
            indf[r] = 0

    indf = indf.loc[:, selected_reservoirs + ["a"]]

    result = pd.DataFrame()

    indf['a'] = indf.a.astype(int)
    indf = indf[indf['a'] > 0] #only positive values of 'a' selected

    dfModel = indf.copy()
    dfModel = coef * dfModel
    dfModel['t'] = indf['a'].apply(lambda x: f"t{int(x):06d}") #converted to string
    dfModel['grid'] = grid
    del dfModel['a']

    #copy data columns to each f
    dataframes = [dfModel.copy() for i in range(0,len(f))]
    for i in range(0,len(f)):
        dataframes[i]['f'] = f[i]
    result = pd.concat(dataframes)
    move = result.pop("grid")
    result.insert(0,"grid",move)

    result.sort_values(['t','f'], inplace=True)
    move = result.pop("f")
    result.insert(1, "f", move )

    move = result.pop("t")
    result.insert(2,"t",move)

    result.reset_index(inplace=True, drop=True)


    #print('\n')
    #print(result.info())
    #print(result.head(10))
        
    result.to_csv('output/bb_ts_influx_hydro.csv',index=False)
    #result.to_excel('summary_inflow_model_form_MWh.xlsx', index=False)

    print("inflow transformation ", round(time.time() - startTime,2), "s  -- done")

    startTime = time.time()
    print('\n')
    print('transforming hydro inflow average data')

    inputfile = "input/summary_hydro_average_year_1982-2020_1h_MWh.csv"
    output10 = "output/bb_ts_influx_hydro_10p.csv"
    output50 = "output/bb_ts_influx_hydro_50p.csv"
    output90 = "output/bb_ts_influx_hydro_90p.csv"

    dfyear = pd.read_csv(inputfile)
    del dfyear['Unnamed: 0']
    f50 = 'f01'
    f10 = 'f02'
    f90 = 'f03'

    #distribute data to all the years
    startyear = pd.to_datetime(timeorigin).year
    endyear = 2020
    leapday = dfyear.tail(24)
    dflist = []
    for year in range(startyear,(endyear+1)):
        dflist.append(dfyear)
        if calendar.isleap(year):
            dflist.append(leapday)
    indf = pd.concat(dflist,axis=0)
    indf.reset_index(drop=True, inplace=True)

    #50%
    dfModel = indf.filter(regex='50_',axis=1).copy()
    #remove prefix from column names
    dfModel.columns = dfModel.columns.str.lstrip('50_')
    # add missing columns as zero to the input data
    for r in selected_reservoirs:
        if not r in dfModel.columns:
            print("reservoir ",r, " not found!")
            dfModel[r] = 0
    #take only selected columns
    dfModel = dfModel.loc[:, selected_reservoirs]
    #transform to model form
    dfModel = coef * dfModel
    dfModel['a'] = dfModel.index + 1
    dfModel.insert(0,"grid",grid)
    dfModel.insert(1, "f", f50)
    dfModel.insert(2,"t",dfModel['a'].apply(lambda x: f"t{int(x):06d}"))
    del dfModel['a']
        
    dfModel.to_csv(output50,index=False)

    #10%
    dfModel = indf.filter(regex='10_',axis=1).copy()
    #remove prefix from column names
    dfModel.columns = dfModel.columns.str.lstrip('10_')
    # add missing columns as zero to the input data
    for r in selected_reservoirs:
        if not r in dfModel.columns:
            print("reservoir ",r, " not found!")
            dfModel[r] = 0
    #take only selected columns
    dfModel = dfModel.loc[:, selected_reservoirs]
    #transform to model form
    dfModel = coef * dfModel
    dfModel['a'] = dfModel.index + 1
    dfModel.insert(0,"grid",grid)
    dfModel.insert(1, "f", f10)
    dfModel.insert(2,"t",dfModel['a'].apply(lambda x: f"t{int(x):06d}"))
    del dfModel['a']

    dfModel.to_csv(output10,index=False)

    
    #90%
    dfModel = indf.filter(regex='90_',axis=1).copy()
    #remove prefix from column names
    dfModel.columns = dfModel.columns.str.lstrip('90_')
    # add missing columns as zero to the input data
    for r in selected_reservoirs:
        if not r in dfModel.columns:
            print("reservoir ",r, " not found!")
            dfModel[r] = 0
    #take only selected columns
    dfModel = dfModel.loc[:, selected_reservoirs]
    #transform to model form
    dfModel = coef * dfModel
    dfModel['a'] = dfModel.index + 1
    dfModel.insert(0,"grid",grid)
    dfModel.insert(1, "f", f90)
    dfModel.insert(2,"t",dfModel['a'].apply(lambda x: f"t{int(x):06d}"))
    del dfModel['a']

    dfModel.to_csv(output90,index=False)
    

    print("average inflow transformation ", round(time.time() - startTime,2), "s  -- done")
    
