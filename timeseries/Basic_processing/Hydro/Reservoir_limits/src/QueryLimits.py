"""
Processing reservoir limits and limits for ps_Open and ps_Closed
Input files: PECD-hydro-weekly-reservoir-levels.csv, PECD-hydro-capacities.csv
Output files: countrycode.csv
"""
from pandas import read_csv
from pandas import DataFrame
import pandas as pd
import time
import os
import sys
from dateutil import parser
from datetime import date,datetime
import numpy as np
from matplotlib import pyplot


class QueryMinMaxLimits:
        """
        Class for creating minmax limits for separate areas
        """

        def __init__(self, ADD=""):
                """
                Initialising variables
                """
                
                self.ADD=ADD

                self.country_codes = [
                'AT00',
                'BE00',	
                'CH00',
                'DE00',
                #'DKW1',
                #'DKE1',
                #'EE00',
                'ES00',
                'FI00',
                'FR00',
                'UK00',
                'LT00',
                'LV00',
                'NL00',
                #'NOS0', #Norway processed with separate python-file
                #'NOM1',
                #'NON1',
                'PL00',
                'SE01',
                'SE02',
                'SE03',
                'SE04',
                'ITN1',
                'ITCN',
                'ITCS',
                'PT00'
                ]


                self.minc = 'Minimum Reservoir levels at beginning of each week (ratio) 0<=x<=1.0'
                self.maxc = 'Maximum Reservoir level at beginning of each week (ratio) 0<=x<=1.0'
                self.start='2015-01-01 00:00:00'
                self.end='2016-12-31 23:00:00'

                #boundarytype
                self.minvariable = 'downwardLimit'
                self.maxvariable = 'upwardLimit'

                # additions to column names
                self.add = '_reservoir'
                self.add_open = '_psOpen'
                self.add_closed = '_psClosed'


        def process_areas(self):
                """
                Create timeseries for each specified area from input file
                """
                startTime = time.time()
                print('\n')
                print('querying PECD min max limits')

                startyear = pd.to_numeric(pd.to_datetime(self.start).year)
                endyear = pd.to_numeric(pd.to_datetime(self.end).year)

                levels_name = os.path.normpath(self.ADD+'input/PECD-hydro-weekly-reservoir-levels.csv')
                capacities_name = os.path.normpath(self.ADD+'input/PECD-hydro-capacities.csv')
                indf = pd.read_csv(levels_name)
                capc = pd.read_csv(capacities_name)
                capc.set_index(["zone","type","variable"], inplace=True)
                indf = indf[indf["year"] <= endyear]
                indf = indf[indf["year"] >= startyear]
                indf["year"] = pd.to_numeric(indf["year"])
                indf["week"] = pd.to_numeric(indf["week"])
                #print(df.head(5))
                #print(indf.dtypes)


                for c in self.country_codes:
                        
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        df1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        dfa = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        dfb = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        print(c)
                        df = indf.copy()

                        # check if limits data has any NaN values
                        serieshasnans = df[df["zone"] == c].iloc[:,3].isna().to_numpy().any()
                        if serieshasnans:
                            print(c + " has Nan values!" )

                        skipdetailedlevels = df[df["zone"] == c].empty or serieshasnans
                        
                        if not skipdetailedlevels:
                                df = df[df["zone"] == c]
                                df.sort_values(by=['year','week'], inplace=True)
                                df.reset_index(inplace=True)
                                for year in range (startyear, (endyear+1)):
                                        yeardf = df[df["year"] == year]
                                        yeardf.reset_index(inplace=True)
                                        fourthday = date(year, 1, 4) + pd.DateOffset(hours=12)
                                        for i in yeardf.index:
                                                t = fourthday + i*pd.DateOffset(days=7)
                                                if (t <= pd.to_datetime(self.end)) and (i <52):
                                                        dfa.at[t,c+self.add]= 1000*yeardf.at[i,self.minc] * capc.at[(c,'Reservoir','Reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                                                        dfb.at[t,c+self.add]= 1000*yeardf.at[i,self.maxc] * capc.at[(c,'Reservoir','Reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                                                else:
                                                        if i==52:
                                                                t = pd.Timestamp(year,12,28) + pd.DateOffset(hours=12)
                                                                dfa.at[t,c+self.add]= 1000*yeardf.at[51,self.minc] * capc.at[(c,'Reservoir','Reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                                                                dfb.at[t,c+self.add]= 1000*yeardf.at[51,self.maxc] * capc.at[(c,'Reservoir','Reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                            
                                dfa.interpolate(inplace=True, limit = 84, limit_direction='both')
                                dfb.interpolate(inplace=True, limit = 84, limit_direction='both')
                        else:
                                print(c + " has no levels")
                                dfa[c+self.add] = 0
                                dfb[c+self.add] = 1000*capc.at[(c,'Reservoir','Reservoir capacity (GWh)'),'value'] #change GWhs to MWhs

                        dfa[c+self.add_open] = 0
                        dfb[c+self.add_open] = 1000*capc.at[(c,'Pump Storage - Open Loop','Cumulated (upper or head) reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                        dfa[c+self.add_closed] = 0
                        dfb[c+self.add_closed] = 1000*capc.at[(c,'Pump Storage - Closed Loop','Cumulated (upper or head) reservoir capacity (GWh)'),'value'] #change GWhs to MWhs

                        dfa['boundarytype'] = self.minvariable
                        dfb['boundarytype'] = self.maxvariable
                        #print(dfa.index)
                        #print(dfb.index)
                        df1h=pd.concat([dfa, dfb])
                        df1h.reset_index(inplace=True)
                        df1h.sort_values(['index','boundarytype'], inplace=True)
                        df1h.set_index(['index','boundarytype'], inplace=True)
                        df1h.rename_axis(["",'boundarytype'], axis="rows", inplace = True)
                        #print(df1h.head(10))
                        df1h.to_csv(csvName)
                               

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        print('\n')

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        limits = QueryMinMaxLimits(ADD)

        limits.process_areas()

