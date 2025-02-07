"""
Transform load file into required model form. Change: grid, timeorigin, timestep, f according to needs
Input file: inputfile
Output file: summary_load_2011_2020_model_form_MWh.xlsx (also.csv with same name)
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
import datetime
import calendar

def conv_loads_tomodelform(selected_regions):

    startTime = time.time()
    print('\n')
    print('transforming load data')

    #Set input values here!
    inputfile = "input/summary_load_2011-2020-1h.csv"
    regionfile = "input/regions.csv"
    grid = "all"
    timeorigin = "1.1.1982 00:00"  
    timestep = 60           #in minutes
    f = ['f00']       #forecast index
    coef = -1           #coefficient with which all the values are multiplied

    # suffix for electricity nodes
    add = '_elec'

    indf = pd.read_csv(inputfile)
    indf['Unnamed: 0'] =  pd.to_datetime(indf['Unnamed: 0'])
    indf['a'] = (indf['Unnamed: 0'] - pd.to_datetime(timeorigin))/ pd.Timedelta(minutes=timestep)+1
    #print(indf['a'].head())

    #region codes
    temp = pd.read_csv(regionfile)
    origregions = dict(temp.values)

    dfModel = pd.DataFrame()
    result = pd.DataFrame()

    indf['a'] = indf.a.astype(int)
    df = indf[indf['a']>0] #only positive values of 'a' selected

    dfModel['t'] = df['a'].apply(lambda x: f"t{x:06d}") #converted to string
    dfModel['grid'] = grid
    #print(dfModel.head())

    for c in selected_regions:
        if c == 'NOS0':
            dfModel['NOS0'+ add] =  coef * (df['NO_1']+df['NO_2']+df['NO_5'] ) #1+2+5
        elif c == 'NOM1':
            dfModel['NOM1'+ add] = coef * df['NO_3'] #3
        elif c == 'NON1':
            dfModel['NON1'+ add] = coef * df['NO_4'] #4
        else:
            #origarea = country_codes[country_codes_final.index(c)] 
            #dfModel[c + add] = coef * df[origarea]
            dfModel[c + add] = coef * df[origregions[c]]

        print(c, " ", round(time.time() - startTime,2), "s  -- done")

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

    result.reset_index(inplace=True, drop=True)

    #print('\n')
    #print(result.info())
    #print(result.head(10))
        
    result.to_csv('output/bb_ts_influx_elec.csv',index=False)
    #result.to_excel('summary_load_2011_2020_model_form_MWh.xlsx', index=False)

    print("transformation ", round(time.time() - startTime,2), "s  -- done")
    
    #########################
    # 10%, 50%, and 90% loads
    #########################
    
    startTime = time.time()
    print('\n')
    print('transforming average load data')

    inputfile = "input/average_load_2011-2020-1h.csv"
    output_prefix = "output/bb_ts_influx_elec"

    dfyear = pd.read_csv(inputfile)
    del dfyear['Unnamed: 0']
    f50 = 'f01'
    f10 = 'f02'
    f90 = 'f03'


    #distribute data to all the years
    startyear = pd.to_datetime(timeorigin).year
    endyear = 2020
    leapday = dfyear[-169:-145]
    dflist = []
    num_mon = 0
    for year in range(startyear,(endyear+1)):
        if year == startyear:
            #add days before first monday
            #number of first Monday of year * 24
            num_mon = ((6-datetime.datetime(year, 1, 1).weekday()+ 1) % 7)
            #copy the same days from week ahead
            if num_mon > 0:
                firstdf = dfyear[((7-num_mon)*24-1): (7*24-1)]
                dflist.append(firstdf)
        if year == endyear:
            #remove num_mon hours from last year
            dfyear = dfyear[:-(num_mon*24)]
        dflist.append(dfyear)
        if calendar.isleap(year):
            dflist.append(leapday)
    df = pd.concat(dflist,axis=0)
    df.reset_index(drop=True, inplace=True)

    # suffix for electricity nodes
    add = '_elec'

    #changing to modelform
    strlist = ['50','10','90']
    flist = ['f01','f02','f03']
    i = 0

    for rate in strlist:

        dfModel = pd.DataFrame()
        result = pd.DataFrame()

        indf = df.filter(regex=(rate+'_'),axis=1).copy()
        indf.columns = indf.columns.str.lstrip(rate+'_')
        for c in selected_regions:
            if c == 'NOS0':
                dfModel['NOS0'+ add] =  coef * (indf['NO_1']+indf['NO_2']+indf['NO_5'] ) #1+2+5
            elif c == 'NOM1':
                dfModel['NOM1'+ add] = coef * indf['NO_3'] #3
            elif c == 'NON1':
                dfModel['NON1'+ add] = coef * indf['NO_4'] #4
            else:
                dfModel[c + add] = coef * indf[origregions[c]]
        for r in selected_regions:
            if not r+add in dfModel.columns:
                print("area ",r, " not found!")
                dfModel[r+add] = 0
        dfModel['a'] = dfModel.index + 1
        dfModel.insert(0,"grid",grid)
        dfModel.insert(1, "f", flist[i])
        i+=1
        dfModel.insert(2,"t",dfModel['a'].apply(lambda x: f"t{int(x):06d}"))
        del dfModel['a']
        #print(dfModel.head())
        #print(dfModel.info())

        filename = output_prefix+'_'+rate+'p.csv'
        dfModel.to_csv(filename,index=False)

    print("average load transformation ", round(time.time() - startTime,2), "s  -- done")


