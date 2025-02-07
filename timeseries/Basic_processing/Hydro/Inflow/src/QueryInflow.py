"""
Generation of inflow values.
Input files: PECD-hydro-weekly-inflows.csv, PECD-hydro-daily-ror-generation.csv
Output files: country_code.csv
"""

from pandas import read_csv
from pandas import DataFrame
import pandas as pd
import time
import os
from datetime import date, datetime
import numpy as np
from matplotlib import pyplot
import calendar


class QueryInflow:
        """
        Class for creating inflows for separate areas
        """

        def __init__(self, ADD=""):
                """
                Initialize variables
                """
                self.ADD=ADD

                self.file_name1 = os.path.normpath(ADD+'input/PECD-hydro-weekly-inflows-corrected.csv')
                self.file_name2 = os.path.normpath(ADD+'input/PECD-hydro-daily-ror-generation.csv')

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
                'NOS0',
                'NOM1',
                'NON1',
                'PL00',
                'SE01',
                'SE02',
                'SE03',
                'SE04'
                #'ITN1',
                #'ITCN',
                #'ITCS',
                #'PT00'
                ]


                self.inflow1 = 'Cumulated inflow into reservoirs per week in GWh'
                self.inflow2 = 'Cumulated NATURAL inflow into the pump-storage reservoirs per week in GWh'
                self.ror = 'Run of River Hydro Generation in GWh per day'
                self.start='1982-01-01 00:00:00'
                self.end='2021-01-01 00:00:00'

                self.add1 = '_reservoir'
                self.add2 = '_psOpen'
                self.add3 = '_ror'


        def process_inflow(self):
                """
                Main functionality
                """
                startTime = time.time()
                print('\n')
                print('querying PECD hydro inflows')
                startyear = pd.to_numeric(pd.to_datetime(self.start).year)
                endyear = pd.to_numeric(pd.to_datetime(self.end).year)
                indf = pd.read_csv(self.file_name1)
                indf2 = pd.read_csv(self.file_name2)

                #use only selected years
                indf = indf[indf["year"] <= endyear]
                indf = indf[indf["year"] >= startyear]
                indf["year"] = pd.to_numeric(indf["year"])
                indf["week"] = pd.to_numeric(indf["week"])
                indf2 = indf2[indf2["year"] <= endyear]
                indf2 = indf2[indf2["year"] >= startyear]
                indf2["year"] = pd.to_numeric(indf2["year"])
                indf2["Day"] = pd.to_numeric(indf2["Day"])


                for c in self.country_codes:

                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        dfinflow1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        #print(c)
                        df = indf.copy()

                        #first handle the inflow file
                        if not df[df["zone"] == c].empty:
                                df = df[df["zone"] == c]
                                df.sort_values(by=['year','week'], inplace=True)
                                df.fillna(0, inplace=True)
                                years = df['year'].drop_duplicates().sort_values()
                                for year in years:
                                        yeardf = df[df["year"] == year]
                                        yeardf.reset_index(inplace=True)
                                        fourthday = date(year, 1, 4) + pd.DateOffset(hours=12)
                                        for i in yeardf.index:
                                                t = fourthday + i*pd.DateOffset(days=7)
                                                if (t <= pd.to_datetime(self.end)) and (i <52):
                                                        dfinflow1h.at[t,c+self.add1]= 1000*yeardf.at[i,self.inflow1]/168 #change GWhs to MWhs, from weekly to average hourly value
                                                        dfinflow1h.at[t,c+self.add2]= 1000*yeardf.at[i,self.inflow2]/168 #change GWhs to MWhs, from weekly to average hourly value
                                                else:
                                                        if i==52:
                                                                t = pd.Timestamp(year,12,28) + pd.DateOffset(hours=12)
                                                                dfinflow1h.at[t,c+self.add1]= 1000*yeardf.at[51,self.inflow1]/168 #change GWhs to MWhs, from weekly to average hourly value
                                                                dfinflow1h.at[t,c+self.add2]= 1000*yeardf.at[51,self.inflow2]/168 #change GWhs to MWhs, from weekly to average hourly value

                                #print(dfinflow1h.info())
                                dfinflow1h[c+self.add1] = dfinflow1h[c+self.add1].interpolate(limit = 84, limit_direction='both')
                                dfinflow1h[c+self.add2] = dfinflow1h[c+self.add2].interpolate(limit = 84,  limit_direction='both')
                        else:
                                print("inflow to reservoir + psOpen null in", c)
                                dfinflow1h.loc[:,c+self.add1] =  np.nan #add null
                                dfinflow1h.loc[:,c+self.add2] =  np.nan #add null
                                
                        #secondly handle the ror file
                        df2 = indf2.copy()
                        if not df2[df2["zone"] == c].empty:
                                df2 = df2[df2["zone"] == c]
                                df2.sort_values(by=['year','Day'], inplace=True)
                                df2.reset_index(inplace=True)
                                df2.fillna(0, inplace=True)
                                years = df2['year'].drop_duplicates().sort_values()
                                for year in years:
                                        yeardf = df2[df2["year"] == year]
                                        yeardf.reset_index(inplace=True)
                                        firstMidday = date(year, 1, 1) + pd.DateOffset(hours=12)
                                        for i in yeardf.index:
                                                t = firstMidday + i*pd.DateOffset(days=1)
                                                if t <= pd.to_datetime(self.end):
                                                        dfinflow1h.at[t,c+self.add3]= 1000*yeardf.at[i,self.ror]/24 #change GWhs to MWhs, from daily to average hourly value
                                        if calendar.isleap(year):
                                                if t < pd.to_datetime(self.end):
                                                        leapt = date(year,12,31) + pd.DateOffset(hours=12)
                                                        dfinflow1h.at[leapt,c+self.add3]= 1000*yeardf.at[364,self.ror]/24 #one additional day for leap years, not in source file
                                #print(dfinflow1h.info())
                                dfinflow1h[c+self.add3] = dfinflow1h[c+self.add3].interpolate(limit = 12, limit_direction='both') 
                        else:
                                print("Ror inflow null in", c)
                                dfinflow1h.loc[:,c+self.add3] = np.nan #add null
                                
                        #print(dfinflow1h.info())
                        #print(dfinflow1h.head(10))

                        dfinflow1h.to_csv(csvName)

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        #print('\n')

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        inflow = QueryInflow(ADD)

        inflow.process_inflow()


