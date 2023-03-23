"""
Processing min/max generation data for Norway as they were not in the original input file
Input files: PEMMDB_NOM1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NON1_Hydro Inflow_SOR 20.xlsx, PEMMDB_NOS0_Hydro Inflow_SOR 20.xlsx 
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
from datetime import date, datetime
import numpy as np
from matplotlib import pyplot


class QueryMinMaxGenerationNorway:
        """
        Class for minmax generation for separate areas
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

                self.minc = 'Minimum Generation in MW'
                self.maxc = 'Maximum Generation in MW'

                self.start='1982-01-01 00:00:00'
                self.end='2021-01-01 00:00:00'

                self.file_first = 'PEMMDB_'
                self.file_last = '_Hydro Inflow_SOR 20.xlsx'

                #boundarytype
                self.minvariable = 'downwardLimit'
                self.maxvariable = 'upwardLimit'
                self.add = '_reservoir'
                
        def process_areas(self):
                """
                Create timeseries for each specified area from input file
                """
                startTime = time.time()
                print('\n')
                print('querying PECD min max generation Norway')

                startyear = pd.to_numeric(pd.to_datetime(self.start).year)
                endyear = pd.to_numeric(pd.to_datetime(self.end).year)


                for c in self.country_codes:
                        
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        df1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        dfa = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        dfb = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        print(c)

                        filename = os.path.normpath(self.ADD+'input/'+self.file_first+c+self.file_last)
                        indf = read_excel(filename,sheet_name='Pump storage - Open Loop',
                                        usecols = "GN:HV",skiprows=12)
                        yearsfloat = [float(x) for x in indf.columns.tolist()]
                        years = [int(x) for x in yearsfloat]
                        indf.rename(columns=dict(zip(indf.columns,years)),inplace=True)
                        #print(indf.head())
                        #print(indf.info())
                        for year in years:
                                fourthday = date(year, 1, 4) + pd.DateOffset(hours=12)
                                df = indf[year]
                                for i in df.index:
                                        t = fourthday + i*pd.DateOffset(days=7)
                                        if t <= pd.to_datetime(self.end) and i < 52:
                                                dfa.at[t,c+self.add]= df[i] # MWhs
                                        else:
                                                if i==52:
                                                        t = pd.Timestamp(year,12,28) + pd.DateOffset(hours=12)
                                                        dfa.at[t,c+self.add]= df[51] # make sure that last year is filled, MWhs

                        dfa.interpolate(inplace=True, limit = 84, limit_direction='both') 
                        dfa['boundarytype'] = self.minvariable
                        dfb[c+self.add]= np.nan # no max variable values
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

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        print('\n')

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        minmaxnorway = QueryMinMaxGenerationNorway(ADD)

        minmaxnorway.process_areas()
