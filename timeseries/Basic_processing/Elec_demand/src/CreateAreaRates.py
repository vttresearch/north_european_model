"""
Area values for DK, NO, and SE missing from year 2011-2014.
The purpose of this is to use the values from 2015-2020 and
generate how sum of area is distributed to areas in non-leap year.
Input file: summary_2015-2020-1h.csv
Output file: rates_for_areas.csv
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import calendar

class AreaRates:
    """
    Class for creating distributon of values in 2011-2014 for areas in
    DK,NO, and SE from distribution of respective values in years 2015-2020
    """

    def __init__(self, ADD=""):
        self.ADD=ADD

        self.country_codes = [
            'DK_1',
            'DK_2',
            'NO_1',
            'NO_2',
            'NO_3',
            'NO_4',
            'NO_5',
            'SE_1',
            'SE_2',
            'SE_3',
            'SE_4'
            ]

        self.years = {
            2015,
            2016,
            2017,
            2018,
            2019,
            2020
            }

        self.rate = '_rate'
        self.csvInput = os.path.normpath(ADD+'output/summary_2015-2020-1h.csv')
            
    def NonLeapAreaRates(self):
        """
        Area rates for non leap years
        """
        startTime = time.time()
        print('\n')
        print('creates rates for areas')
        
        df = pd.read_csv(self.csvInput)
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], infer_datetime_format=True)
        df.set_index('Unnamed: 0', inplace=True)
        #print(df.head(5))
        rates = pd.DataFrame()

        for c in self.country_codes:
            df[c+self.rate] = df[c].div(df[c[0:2]],axis='index')
            x = pd.DataFrame()
            for y in self.years:
                #reorganise rates by year
                if calendar.isleap(y):
                    df = df[(pd.DatetimeIndex(df.index).month != 2) | (pd.DatetimeIndex(df.index).day != 29)] 
                Onecolumn = df[pd.DatetimeIndex(df.index).year == y][c+self.rate]
                x[c,y] = pd.DataFrame(Onecolumn.values) 
            #calculate mean of year rates
            rates[c] = x.mean(axis=1)

        #print(rates.head(5))
        #print(rates.describe())
        outname = os.path.normpath(self.ADD+'output/rates_for_areas.csv')
        rates.to_csv(outname)

        print("rates calculated ", round(time.time() - startTime,2), "s  -- done")
        return

    def LeapAreaRates(self):
        """
        Area rates for leap years
        """
        startTime = time.time()
        print('\n')
        print('creates leap year rates for areas')

        df = pd.read_csv(self.csvInput)
        df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], infer_datetime_format=True)
        df.set_index('Unnamed: 0', inplace=True)
        #print(df.head(5))
        rates = pd.DataFrame()

        for c in self.country_codes:
            df[c+self.rate] = df[c].div(df[c[0:2]],axis='index')
            x = pd.DataFrame()
            for y in self.years:
                #reorganise rates by year
                if calendar.isleap(y):
                    Onecolumn = df[pd.DatetimeIndex(df.index).year == y][c+self.rate]
                    x[c,y] = pd.DataFrame(Onecolumn.values)
            #calculate mean of year rates
            rates[c] = x.mean(axis=1)

        #print(rates.head(5))
        #print(rates.describe())

        outname = os.path.normpath(self.ADD+'output/rates_for_areas_leap_year.csv')
        rates.to_csv(outname)

        print("leap rates calculated ", round(time.time() - startTime,2), "s  -- done")
        

"""
    Used in testing
    
"""

if __name__ == "__main__":
    ADD = "../"

    rates = AreaRates(ADD)
        
    rates.NonLeapAreaRates()
    rates.LeapAreaRates()



