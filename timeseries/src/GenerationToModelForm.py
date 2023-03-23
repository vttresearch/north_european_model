"""
Transform levels file into required model form.
Change: grid, begintime, endtime, timestep, f, coefficient, maxval, and areas according to needs
Input file: summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv.csv
Output file: summary_reservoir_minmax_generation_model_form_MWh.xlsx (also.csv with same name)
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime
import calendar

def conv_generation_tomodelform(selected_regions):

    startTime = time.time()
    print('\n')
    print('transforming min and max generation data')

    #Set input values here!
    inputfilename = "input/summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv"
    outputfilename = "output/bb_ts_influx_genlim.csv"
    reservoirfile = "input/region_reservoir.csv"
    writemingen = True
    writemaxgen = False
    grid = "all"
    begintime = "1.1.1982 00:00"  
    timestep = 60             #in minutes
    endtime = "1.1.2018"
    f = ["f00"]
    coef = -1.0   #coefficient with which all the values are multiplied
    maxval = 30000 #user defined value for upward limit when missing
    minval = 0   #default value, when no values in downward limit

    # select specific reservoirs
    region_reservoir = pd.read_csv(reservoirfile)
    region_reservoir = region_reservoir[region_reservoir["region"].isin(selected_regions)]
    selected_reservoirs = region_reservoir["reservoir"].values.tolist()
    print(selected_reservoirs + ["a"])

    countries = [
    #'AT00_reservoir',
    #'BE00_reservoir',	
    #'CH00_reservoir',
    #'DE00_reservoir',
    #'DKW1_reservoir',
    #'DKE1_reservoir',
    #'EE00_reservoir',
    #'ES00_reservoir',
    'FI00_reservoir',
    #'FR00_reservoir',
    #'UK00_reservoir',
    #'LT00_reservoir',
    #'LV00_reservoir',
    #'NL00_reservoir',
    #'NOS0_psOpen',
    #'NOM1_psOpen',
    #'NON1_psOpen',
    #'PL00_reservoir',
    'SE01_reservoir',
    'SE02_reservoir',
    'SE03_reservoir',
    'SE04_reservoir'
    ]



    minvariable = 'downwardLimit'
    maxvariable = 'upwardLimit'
    add_min = '_mingen'
    add_max = '_maxgen'

    start = pd.to_datetime(begintime)
    end = pd.to_datetime(endtime)

    # initialise 
    indf = pd.read_csv(inputfilename)
    indf.set_index(['Unnamed: 0'], inplace=True)
    df = pd.DataFrame(index=indf.index.copy())

    # only selected areas added, nans when missing values
    for c in selected_reservoirs:
            print(c)
            try:
                    indf = pd.read_csv(inputfilename, usecols=['Unnamed: 0','boundarytype',c])
            except:
                    if writemingen == True: df[c+add_min] = minval
                    if writemaxgen == True: df[c+add_max] = maxval
                    print("reservoir ",c, " not found.")
            else:
                    #copy downward limit
                    if writemingen == True:
                        indf2 = indf[indf['boundarytype'] == minvariable].copy()
                        indf2.set_index(['Unnamed: 0'], inplace=True)
                        del indf2['boundarytype']
                        indf2.fillna(minval, inplace=True)
                        df[c+add_min] = indf2[c]
                    #copy upward limit
                    if writemaxgen == True:
                        indf2 = indf[indf['boundarytype'] == maxvariable].copy()
                        indf2.set_index(['Unnamed: 0'], inplace=True)
                        del indf2['boundarytype']
                        indf2.fillna(maxval, inplace=True)
                        df[c+add_max] = indf2[c]

    # select timeframe
    df.reset_index(inplace=True)
    df.drop_duplicates(inplace=True)
    df['Unnamed: 0'] =  pd.to_datetime(df['Unnamed: 0'])
    df = df[df['Unnamed: 0']<= end]
    df['a'] = (df['Unnamed: 0'] - start)/ pd.Timedelta(minutes=timestep)+1

    dfModel = pd.DataFrame()
    result = pd.DataFrame()
    df['a'] = df.a.astype(int)
    dfModel = df[df['a']>0].copy() #only positive values of 'a' selected

    # multiply with coefficient
    dfModel.set_index(['Unnamed: 0','a'], inplace=True)
    dfModel = coef * dfModel
    dfModel.reset_index(inplace=True)

    #add t converted backbone time format and grid
    dfModel['t'] = dfModel['a'].apply(lambda x: f"t{x:06d}") 
    dfModel['grid'] = grid

    #remove irrelevant columns
    del dfModel['Unnamed: 0']
    del dfModel['a']


    #copy data columns to each f
    dataframes = [dfModel.copy() for i in range(0,len(f))]
    for i in range(0,len(f)):
        dataframes[i]['f'] = f[i]
    result = pd.concat(dataframes)

    result.sort_values(['t','f'], inplace=True)

    move = result.pop("grid")
    result.insert(0,"grid",move)

    move = result.pop("f")
    result.insert(1, "f", move )

    move = result.pop("t")
    result.insert(2,"t",move)

    result.reset_index(inplace=True, drop=True)

    print('\n')
    print(result.info())
    print(result.head(10))
    print(result.tail(10))
        
    result.to_csv(outputfilename, index=False)
    #result.to_excel('summary_reservoir_minmax_generation_model_form_MWh.xlsx', index=False)

    print("transformation ", round(time.time() - startTime,2), "s  -- done")
