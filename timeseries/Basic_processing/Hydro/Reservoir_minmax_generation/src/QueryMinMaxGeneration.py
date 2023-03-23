"""
Processing min/max generation data
Input files: PECD-hydro-weekly-reservoir-min-max-generation.csv
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

class QueryMinMaxGeneration:
        """
        Class for minmax generation for separate areas
        """

        def __init__(self, ADD=""):
                """
                Initialising variables
                """
                
                self.ADD=ADD

                self.minc = 'Minimum Generation in MW'
                self.maxc = 'Maximum Generation in MW'
                self.start='1982-01-01 00:00:00'
                self.end='2021-01-01 00:00:00'


                self.country_codes = [
                'AT00',
                'BE00',	
                'CH00',
                'DE00',
                'DKW1',
                'DKE1',
                'EE00',
                'ES00',
                'FI00',
                'FR00',
                'UK00',
                'LT00',
                'LV00',
                'NL00',
                'NOS0',
                'NOM1',
                'NON1',
                'PL00',
                'SE01',
                'SE02',
                'SE03',
                'SE04'
                ]


                self.minvariable = 'downwardLimit'
                self.maxvariable = 'upwardLimit'
                self.add = '_reservoir'

                self.input_file = os.path.normpath(ADD+'input/PECD-hydro-weekly-reservoir-min-max-generation.csv')

        def process_areas(self):
                """
                Create timeseries for each specified area from input file
                """
                startTime = time.time()
                print('\n')
                print('querying PECD min max generation')


                startyear = pd.to_numeric(pd.to_datetime(self.start).year)
                endyear = pd.to_numeric(pd.to_datetime(self.end).year)

                indf = pd.read_csv(self.input_file)
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
                        #df.plot()
                        #pyplot.show()

                        df = indf.copy()
                        
                        if not df[df["zone"] == c].empty:
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
                                                        dfa.at[t,c+self.add]= yeardf.at[i,self.minc] #MWhs
                                                        dfb.at[t,c+self.add]= yeardf.at[i,self.maxc] #MWhs
                                                else:
                                                        if i==52:
                                                                t = pd.Timestamp(year,12,28) + pd.DateOffset(hours=12)
                                                                dfa.at[t,c+self.add]= yeardf.at[51,self.minc] #make sure that last year is filled, MWhs
                                                                dfb.at[t,c+self.add]= yeardf.at[51,self.maxc] #make sure that last year is filled, MWhs
                                dfa.interpolate(inplace=True, limit = 84, limit_direction='both') 
                                dfb.interpolate(inplace=True, limit = 84, limit_direction='both') 
                                dfa['boundarytype'] = self.minvariable
                                dfb['boundarytype'] = self.maxvariable
                                #print(dfa.index)
                                #print(dfb.index)
                                df1h=pd.concat([dfa, dfb])
                                df1h.reset_index(inplace=True)
                                df1h.sort_values(['index','boundarytype'], inplace=True)
                                
                                df1h.set_index(['index','boundarytype'], inplace=True)
                                df1h.rename_axis(["",'boundarytype'], axis="rows", inplace = True)
                                #print(df1h.head(6))
                                #print(df1h.info())
                                df1h.to_csv(csvName)

                                
                        else:
                                print("no values")
                        

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        print('\n')

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        minmax = QueryMinMaxGeneration(ADD)

        minmax.process_areas()
