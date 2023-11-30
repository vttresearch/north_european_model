"""
Create average year of VRE 
Input files: PECD-MAF2019-PV_byarea.csv, PECD-MAF2019-WindOffshore_byarea.csv, PECD-MAF2019-WindOnshore_byarea.csv
Output file: 
"""
from pandas import read_csv
from pandas import DataFrame
import pandas as pd
import time
import os
import sys
import numpy as np
import calendar

def conv_AverageVRE(timeorigin, selected_regions):
        """
        Average year of VRE data
        """
        
        startTime = time.time()
        print('\n')
        print('Processing average year of VRE')

        #Set input values here!
        inPV = "input/vre/PECD-MAF2019-PV_byarea.csv"
        inoffshorewind = "input/vre/PECD-MAF2019-WindOffshore_byarea.csv"
        inonshorewind = "input/vre/PECD-MAF2019-WindOnshore_byarea.csv"
        grid = "all"
        #timeorigin = "1.1.1982 00:00"  
        timestep = 60             #in minutes
        f = ['f00']
        coef = 1.0   #coefficient with which all the values are multiplied
        suffix = '_elec'

        flowlist = ['onshorewind','PV','offshorewind']
        inlist = [inonshorewind,inPV,inoffshorewind]
        d = {'flow' : flowlist,
             'infile':inlist}
        infiles = pd.DataFrame.from_dict(d,orient='index').transpose()
        infiles.set_index('flow',inplace=True)
        dflist = []

        for d in flowlist:

                dfyear = pd.DataFrame()
                indf = pd.read_csv(infiles.at[d,'infile'])
                indf['time'] = pd.to_datetime(indf['time'])

                #leap year processing
                indf['is_leap'] = indf['time'].dt.year.apply(lambda e: calendar.isleap(e))
                indf.drop(indf.loc[(indf['is_leap'])&(indf['time'].dt.month == 12)&(indf['time'].dt.day == 31)].index, inplace=True)
                indf.reset_index(drop = True, inplace=True)
                indf['group'] = indf.index % 8760
                del indf['time']
                del indf['is_leap']

                for c in selected_regions:
                        avg_year = pd.DataFrame()

                        #group data in 8760 rows and take average of each row
                        if (not indf[c].dropna().empty) and (indf[c].sum()!=0):
                                avg_year = indf.groupby([indf.group], as_index = False)[c].quantile(0.1)
                                dfyear['10_'+c+suffix] = avg_year[c]
                                avg_year = indf.groupby([indf.group], as_index = False)[c].quantile(0.5)
                                dfyear['50_'+c+suffix] = avg_year[c]
                                avg_year = indf.groupby([indf.group], as_index = False)[c].quantile(0.9)
                                dfyear['90_'+c+suffix] = avg_year[c]
                                #dfyear[['10_'+c+suffix,'50_'+c+suffix,'90_'+c+suffix]].plot()
                                #pyplot.show()
                dfyear['flow'] = d
                #print(dfyear.info())
                dflist.append(dfyear)


        ##########################
        #Transofrm into model-form
        ##########################
        print('Processing modelform of average VRE')
        
        output_prefix = "output/bb_ts_cf_io"
        startyear = pd.to_datetime(timeorigin).year
        endyear = 2020
        tslist = []
        strlist = ['50','10','90']
        flist = ['f01','f02','f03']
        
        #distribute data to all the years
        for df in dflist:
                leapday = df.tail(24)
                yearlist = []
                for year in range(startyear,(endyear+1)):
                        yearlist.append(df)
                        if calendar.isleap(year):
                                yearlist.append(leapday)
                indf = pd.concat(yearlist,axis=0)
                indf.reset_index(drop=True, inplace=True)
                tslist.append(indf)

        i = 0

        for rate in strlist:

                print('\n')
                print(f"Processing percentile " + rate)

                modellist = []
                for tsdf in tslist:

                        dfModel = pd.DataFrame()
                        indf = tsdf.filter(regex=(rate+'_'),axis=1).copy()
                        indf.columns = indf.columns.str.lstrip(rate+'_')
                        dfModel = coef * indf
                        dfModel['a'] = tsdf.index+1
                        dfModel['flow'] = tsdf['flow']
                        for r in selected_regions:
                                if not r+suffix in dfModel.columns:
                                        print("area ",r, " not found!")
                                        dfModel[r+suffix] = 0
                        #print(dfModel.head())
                        modellist.append(dfModel)
                        
                result = pd.concat(modellist,axis=0)
                move = result.pop('flow')
                result.insert(0,'flow',move)
                result.insert(1, "f", flist[i])
                i+=1
                result.insert(2,"t",result['a'].apply(lambda x: f"t{int(x):06d}"))
                del result['a']
                #print(result.info())

                # Add rounding step for the entire dataframe
                result = result.round(decimals=4) 

                filename = output_prefix+'_'+rate+'p.csv'
                result.to_csv(filename,index=False)

        print('\n')
        print(round(time.time() - startTime,2), "s  -- VRE transformations done")

