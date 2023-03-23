"""
Processing reservoir limits for Norway as they were not in the original input file, psClosed included from capacity
Input files: PEMMDB_NOM1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NON1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NOS0_Hydro Inflow_SOR 20.xlsx, PECD-hydro-capacities.csv 
Output files: countrycode.csv
"""

from pandas import read_csv
from pandas import read_excel
from pandas import DataFrame
import pandas as pd
import time
import os
import sys
from dateutil import parser
from datetime import date,datetime
import numpy as np
from matplotlib import pyplot


class QueryMinMaxLimitsNorway:
        """
        Class for creating minmax limits for separate areas
        """

        def __init__(self, ADD=""):
                """
                Initialising variables
                """
                
                self.ADD=ADD


                self.country_codes = [
                'NOM1',
                'NON1',
                'NOS0'
                ]

                self.minc = 'Minimum Reservoir levels at beginning of each week'
                self.maxc = 'Maximum Reservoir level at beginning of each week'

                self.start='2015-01-01 00:00:00'
                self.end='2016-12-31 23:00:00'

                self.file_first = 'PEMMDB_'
                self.file_last = '_Hydro Inflow_SOR 20.xlsx'

                #boundarytype
                self.minvariable = 'downwardLimit'
                self.maxvariable = 'upwardLimit'

                #column names
                self.add = '_psOpen'
                self.add_closed = '_psClosed'

        def process_areas(self):
                """
                Create timeseries for each specified area from input file
                """
                startTime = time.time()
                print('\n')
                print('querying PECD min max limits Norway')

                capacities_name = os.path.normpath(self.ADD+'input/PECD-hydro-capacities.csv')
                capc = pd.read_csv(capacities_name)
                capc.set_index(["zone","type","variable"], inplace=True)
                #print(df.head(5))
                #print(indf.dtypes)

                startyear = pd.to_numeric(pd.to_datetime(self.start).year)
                endyear = pd.to_numeric(pd.to_datetime(self.end).year)

                for c in self.country_codes:
                        
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        df1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        dfa = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        dfb = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        print(c)

                        filename = os.path.normpath(self.ADD+'input/'+self.file_first+c+self.file_last)
                        df = read_excel(filename,sheet_name='Pump storage - Open Loop',
                                        usecols = "L,M", names=[self.minc, self.maxc],skiprows=12)
                        #print(df.info())
                        #print(df.head())
                        for year in range(startyear,(endyear+1)):
                                fourthday = date(year, 1, 4) + pd.DateOffset(hours=12)
                                for i in df.index:
                                        t = fourthday + i*pd.DateOffset(days=7)
                                        if (t <= pd.to_datetime(self.end)) and (i <52) :
                                                dfa.at[t,c+self.add]= 1000*df.at[i,self.minc] * capc.at[(c,'Pump Storage - Open Loop','Cumulated (upper or head) reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                                                dfb.at[t,c+self.add]= 1000*df.at[i,self.maxc] * capc.at[(c,'Pump Storage - Open Loop','Cumulated (upper or head) reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                                        else:
                                                if i==52:
                                                        t = pd.Timestamp(year,12,28) + pd.DateOffset(hours=12)
                                                        dfa.at[t,c+self.add]= 1000*df.at[51,self.minc] * capc.at[(c,'Pump Storage - Open Loop','Cumulated (upper or head) reservoir capacity (GWh)'),'value'] #change GWhs to MWhs
                                                        dfb.at[t,c+self.add]= 1000*df.at[51,self.maxc] * capc.at[(c,'Pump Storage - Open Loop','Cumulated (upper or head) reservoir capacity (GWh)'),'value'] #change GWhs to MWhs

                        dfa.interpolate(inplace=True, limit = 84, limit_direction='both')
                        dfb.interpolate(inplace=True, limit = 84, limit_direction='both')
                        #add _psClosed levels
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
                        #df1h.plot()
                        #pyplot.show()
                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        print('\n')


"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        limits = QueryMinMaxLimitsNorway(ADD)

        limits.process_areas()

