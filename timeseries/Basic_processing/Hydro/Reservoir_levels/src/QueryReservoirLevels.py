"""
Generation of historical reservoir levels from hydro entsoe transparency data.
Input files: ts_reservoir_level.csv
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


class QueryLevels:
        """
        Class for creating hourly values for historical reservoir levels
        """

        def __init__(self, ADD=""):
                """
                Initialize variables
                """
                self.ADD=ADD

                self.file_name = os.path.normpath(ADD+'input/ts_reservoir_level.csv')

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
                'NO',
                'NO',
                'PL00',
                'SE01',
                'SE02',
                'SE03',
                'SE04'
                ]

                self.original_codes = [
                'AT',
                'BE',	
                'CH',
                'DE',
                'DK_1',
                'DK_2',
                'EE',
                'ES',
                'FI',
                'FR',
                'GB',
                'LT',
                'LV',
                'NL',
                'NO_1',
                'NO_2',
                'NO_3',
                'NO_4',
                'NO_5',
                'PL',
                'SE_1',
                'SE_2',
                'SE_3',
                'SE_4'
                ]

                self.start='1982-01-01 00:00:00'
                self.end='2021-01-01 00:00:00'


        def process_levels(self):
                """
                Interpolate weekly values into hourly values and extrapolate missing years
                """
                startTime = time.time()
                print('\n')
                print('querying ENTSOE transparency historical reservoir levels')
                
                startyear = pd.to_numeric(pd.to_datetime(self.start).year)
                startweek = pd.to_numeric(pd.to_datetime(self.start).week)
                # storing values to be interpolated starts from what is the fourth day of the starting year
                startfourthday = date(startyear, 1, 4) + pd.DateOffset(hours=12)

                indf = pd.read_csv(self.file_name)
                indf['time']= pd.to_datetime(indf['time'])
                datefirst = indf.at[0,'time']
                indf.set_index('time', inplace=True)
                indf["week"] = pd.DatetimeIndex.isocalendar(indf.index).week
                indf["year"] = pd.DatetimeIndex.isocalendar(indf.index).year
                years = sorted(indf['year'].unique())
                #print(indf.info())
                #print(indf.head())
                early = pd.DataFrame(index = pd.date_range(self.start, datefirst, freq='60 min'))
                weeks = int(len(early)/168) #number of weeks at the most

                for c in self.original_codes:
                         
                        try:
                                temp = indf[c].copy()                                
                        except:
                                print(c)
                                print("Area not found")
                                print(c, " ", round(time.time() - startTime,2), "s  -- done")
                                print('\n')
                        else:                                
                                # getting a "typical year" for extrapolation
                                # first transform into df where each input year is one column
                                Onecolumn = pd.DataFrame(index=(1,53))
                                normal = pd.DataFrame()
                                dflist = list()
                                for y in years:
                                        Onecolumn = indf.loc[indf['year'] == y][['week',c]]
                                        Onecolumn.set_index('week',inplace=True)
                                        Onecolumn.rename(columns={c:y}, inplace=True)
                                        dflist.append(Onecolumn)
                                x = pd.concat(dflist, axis=1)
                                #calculate mean of week values
                                normal[c] = x.mean(axis=1)
                                # create extrapolated df using typical year where there is no measures
                                # Note: currently it is assumed that extrapolation is used only for years older than measured values
                                for i in range (weeks):
                                        t = startfourthday + i*pd.DateOffset(days=7)                                         
                                        early.at[t,c] = normal.at[t.week,c]
                                
                # delete temporary columns
                del indf['week']
                del indf['year']
                #drop nans 
                early.dropna(inplace=True)
                # combine extrapolated and mesured dfs
                alldf = pd.concat([early, indf])
                areas = list(alldf.columns.values) #list of areas that have measured values
                #print(areas)

                for c in areas:
                        df1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                        print(c)
                        # change codes, Norway is exception
                        if c[:2] == 'NO':
                                nor = pd.DataFrame()
                                #temp = nor.copy()
                                if c == 'NO_1':
                                        title = 'NOS0'
                                        nor[title] = alldf['NO_1']+alldf['NO_2']+alldf['NO_5'] #1+2+5
                                        temp = nor[title].copy()
                                else:
                                        if c == 'NO_3':
                                                title = 'NOM1'
                                                nor[title] = alldf['NO_3'] #3
                                                temp = nor[title].copy()
                                        else:
                                                if c == 'NO_4':
                                                        title = 'NON1'
                                                        nor[title] = alldf['NO_4'] #4
                                                        temp = nor[title].copy()
                                temp.dropna(inplace=True)
                        else:
                                temp = alldf[c].copy()
                                temp.dropna(inplace=True)
                                title = self.country_codes[self.original_codes.index(c)]
                        if not temp.empty:        
                                for i in range(len(temp)):
                                        t = startfourthday + i*pd.DateOffset(days=7) 
                                        df1h.at[t,title] = temp[i]

                                # interpolate between weekly values to great hourly timeseries
                                df1h.interpolate(inplace=True, limit = 84, limit_direction='both')
                                df1h = df1h.loc[self.start : self.end]
                                #print("Number of nulls in df: ", df1h.isnull().sum())
                                # convert all values to int
                                df1h=df1h.astype(int)
                                #print(df1h.info())
                                #print(df1h.head())
                                        
                                csvName = os.path.normpath(self.ADD+'output/'+ title +'.csv')                                
                                df1h.to_csv(csvName)


                                print(c, " ", round(time.time() - startTime,2), "s  -- done")
                                print('\n')

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        levels = QueryLevels(ADD)

        levels.process_levels()


