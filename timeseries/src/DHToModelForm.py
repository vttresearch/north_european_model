"""
Transform load file into required model form. Change: grid, timeorigin, timestep, f, coef according to needs
Input file: DH_2025_timeseries_summary.csv, DH_2030_timeseries_summary.csv
Output file: DH_2025_summary_model_form.xlsx, DH_2030_summary_model_form.xlsx (also.csv with same name)

Available country_codes:
'AT00',
'DE00',
'DKW1',
'DKE1',
'FI00',
'Helsinki',
'Espoo',
'Vantaa',
'Turku',
'Tampere',
'Oulu',
'Jyväskylä',
'FI00_others',
'LT00',
'LV00',
'NOS0',
'NOM1',
'NON1',
'PL00',
'SE01',
'SE02',
'SE03',
'SE04'

"""

import pandas as pd
import numpy as np
import time
import os
import sys
import argparse
import datetime
import calendar

def conv_dhdemand_tomodelform(selected_regions):

    startTime = time.time()
    print('\n')
    print('transforming DH data')

    #Set input values here!
    output_prefix = "output/bb_ts_influx_heat"
    grid = "all"
    timeorigin = "1.1.1982 00:00"  
    timestep = 60             #in minutes
    f = ['f00']
    coef = -1   #coefficient with which all the values are multiplied

    # suffix for the DH node model name
    suffix = '_dheat'
    yearlist = ['2025','2030']

    for yt in yearlist:    
        inputfile = "input/DH_"+yt+"_timeseries_summary.csv"
        inputaverage = "input/DH_"+yt+"_timeseries_average.csv"
        outputfile = output_prefix+"_"+yt+".csv"
        print('\n')
        print(yt)

        #year values transformed
        indf = pd.read_csv(inputfile)
        indf['Unnamed: 0'] =  pd.to_datetime(indf['Unnamed: 0'])
        indf['a'] = (indf['Unnamed: 0'] - pd.to_datetime(timeorigin))/ pd.Timedelta(minutes=timestep)+1
        indf['a'] = indf.a.astype(int)
        temp_25 = indf[indf['a']>0] #only positive values of 'a' selected
        dfModel = pd.DataFrame()
        dfModel['t'] = temp_25['a'].apply(lambda x: f"t{x:06d}") #converted to string
        dfModel['grid'] = grid
        for c in selected_regions:
            title = c + suffix
            dfModel[title] = coef * indf[title]

        #copy data columns to each f
        dataframes = [dfModel.copy() for i in range(0,len(f))]
        for i in range(0,len(f)):
            dataframes[i]['f'] = f[i]
        df = pd.concat(dataframes)
        move = df.pop("grid")
        df.insert(0,"grid",move)

        df.sort_values(['t','f'], inplace=True)
        move = df.pop("f")
        df.insert(1, "f", move )

        df.reset_index(inplace=True, drop=True)
        print('\n')
        print(df.info())
        print(df.head(10))
            
        df.to_csv(outputfile, index=False)

        print("transformation ", round(time.time() - startTime,2), "s  -- done")

        #########################
        #10%,50%,90% model format
        #########################

        startTime = time.time()
        print('\n')
        print('transforming average DH data')

        dfyear = pd.read_csv(inputaverage)
        del dfyear['Unnamed: 0']

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

        # suffix for DH nodes
        add = '_dheat'

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
                title = c + suffix
                dfModel[title] = coef * indf[title]
            dfModel['a'] = dfModel.index + 1
            dfModel.insert(0,"grid",grid)
            dfModel.insert(1, "f", flist[i])
            i+=1
            dfModel.insert(2,"t",dfModel['a'].apply(lambda x: f"t{int(x):06d}"))
            del dfModel['a']
            print(dfModel.head())
            print(dfModel.info())

            filename = output_prefix+'_'+yt+'_'+rate+'p.csv'
            dfModel.to_csv(filename,index=False)

        print("average transformation ", round(time.time() - startTime,2), "s  -- done")
        print('\n')


