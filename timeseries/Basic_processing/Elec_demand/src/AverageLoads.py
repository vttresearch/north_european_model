
"""
10%,50%,90% loads for a year.
Input files: summary_load_2011-2020-1h.csv
Output file: average_load_2011-2020-1h.csv
"""


import pandas as pd
from pandas import read_excel
import numpy as np
import time
import os
import sys
import datetime
from matplotlib import pyplot
import calendar

class AverageLoads:
    """
    Class for creating 10%,50%,90% loads 
    """

    def __init__(self, ADD="", scenario = "", year = 0):
        
        self.ADD=ADD
        self.scenario = scenario
        self.year = year

        self.country_codes = [
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


        #files to be combined
        self.csvInput = os.path.normpath(self.ADD+'output/summary_load_2011-2020-1h.csv')


    def process_average(self):
        startTime = time.time()
        print('\n')
        print('creating average year from 2011-2020')
        
        indf = pd.read_csv(self.csvInput)
        indf['Date'] = pd.to_datetime(indf['Unnamed: 0'])
        del indf['Unnamed: 0']

        #Note: it is assumed that first row of indf starts from 1st of Jan!
    
        #create group numbers for each hour in a year, starting on first Monday
        firstyear = indf['Date'].iloc[0].year
        if indf['Date'].iloc[0] != datetime.datetime(firstyear, 1, 1):
            print("Potential problem: input file does not start from start of year\n")
        lastyear = indf['Date'].iloc[-1].year
        indf['group'] = 0
        start= 0
        end = 0
        #number of first Monday of year * 24
        num_mon = ((6-datetime.datetime(firstyear, 1, 1).weekday()+ 1) % 7)* 24
        end = num_mon- 1
        #print(num_mon)
        for year in range(firstyear,(lastyear+1)):
            if calendar.isleap(year):
                modyear = 8760 + 24 
            else:
                modyear = 8760
            #counter of hours in a year
            mylist = list(range(0,modyear))
            start = end + 1
            end += modyear
            if (end+1) < indf.shape[0]:
                #assign group numbers to hours in a year
                indf.loc[start:end,'group'] = mylist
            else:
                #the last year is shorter because of starting each year on Monday
                newend = indf.shape[0] - 1
                diff = end - newend
                mylist = mylist[0:(modyear-diff)]
                indf.loc[start:newend,'group'] = mylist
        print(indf.info())
        del indf['Date']
        dfinflow1h = pd.DataFrame()

        for c in self.country_codes:
            if (not indf[c].dropna().empty) and (indf[c].sum()!=0):
                #group by group numbers
                #NOTE: switching the content of 10p and 90p that 10p means low production, but high demand
                avg_df = indf.groupby([indf.group], as_index = False)[c].quantile(0.9)
                dfinflow1h['10_'+c] = avg_df[c]
                avg_df = indf.groupby([indf.group], as_index = False)[c].quantile(0.5)
                dfinflow1h['50_'+c] = avg_df[c]
                avg_df = indf.groupby([indf.group], as_index = False)[c].quantile(0.1)
                dfinflow1h['90_'+c] = avg_df[c]
                
                #dfinflow1h[['10_'+c,'50_'+c,'90_'+c]].plot()
                #pyplot.show()

        #only 8760 rows are used i.e. non-leap year
        dfinflow1h.drop(dfinflow1h.tail(24).index,inplace=True)
        
        #rounding values to int
        result = dfinflow1h.round(0)
        result = result.convert_dtypes()
        print(result.info())

        
        csvOutput1h = os.path.normpath(self.ADD+'output/average_load_2011-2020-1h.csv')
        result.to_csv(csvOutput1h)


        print("load averages ", round(time.time() - startTime,2), "s  -- done")

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        averages = AverageLoads(ADD)        
        averages.process_average()
