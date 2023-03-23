"""
Combine individual DH files into one file 
Input files: DH_2025_timeseries_summary.csv, DH_2030_timeseries_summary.csv
Output file: DH_2025_timeseries_average.csv, DH_2030_timeseries_average.csv
"""
from pandas import read_csv
from pandas import DataFrame
import pandas as pd
import time
import os
import sys
import datetime
import numpy as np
from matplotlib import pyplot
import calendar

class DHAverage:
        """
        Class for averaging separate areas
        """

        def __init__(self, ADD=""):
                """
                Initialising variables
                """
                
                self.ADD=ADD


                self.country_codes = [
                'AT00_dheat',
                'DE00_dheat',
                'DKW1_dheat',
                'DKE1_dheat',
                'EE00_dheat',
                'FI00_dheat',
                'Helsinki_dheat',
                'Espoo_dheat',
                'Vantaa_dheat',
                'Turku_dheat',
                'Tampere_dheat',
                'Oulu_dheat',
                'Jyvaskyla_dheat',
                'FI00_others_dheat',
                'LT00_dheat',
                'LV00_dheat',
                'NOS0_dheat',
                'NOM1_dheat',
                'NON1_dheat',
                'PL00_dheat',
                'SE01_dheat',
                'SE02_dheat',
                'SE03_dheat',
                'SE04_dheat'
                ]

                self.outName_25 = os.path.normpath(ADD+'output/DH_2025_timeseries_average.csv')
                self.outName_30 = os.path.normpath(ADD+'output/DH_2030_timeseries_average.csv')
                self.inName_25 = os.path.normpath(ADD+'output/DH_2025_timeseries_summary.csv')
                self.inName_30 = os.path.normpath(ADD+'output/DH_2030_timeseries_summary.csv')

        def process_average(self):
                """
                Create 10%,50%,90% timeseries for a year 
                """
                startTime = time.time()
                print('\n')
                print('averaging DH production')

                ####
                25
                ####

                indf = pd.read_csv(self.inName_25)
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
                #print(indf.info())
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
        
                result.to_csv(self.outName_25)

                ###
                30
                ###
                
                indf = pd.read_csv(self.inName_30)
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
                #print(indf.info())
                del indf['Date']
                dfinflow1h = pd.DataFrame()

                for c in self.country_codes:
                    if (not indf[c].dropna().empty) and (indf[c].sum()!=0):
                        #group by group numbers
                        avg_df = indf.groupby([indf.group], as_index = False)[c].quantile(0.1)
                        dfinflow1h['10_'+c] = avg_df[c]
                        avg_df = indf.groupby([indf.group], as_index = False)[c].quantile(0.5)
                        dfinflow1h['50_'+c] = avg_df[c]
                        avg_df = indf.groupby([indf.group], as_index = False)[c].quantile(0.9)
                        dfinflow1h['90_'+c] = avg_df[c]
                
                        #dfinflow1h[['10_'+c,'50_'+c,'90_'+c]].plot()
                        #pyplot.show()

                #only 8760 rows are used i.e. non-leap year
                dfinflow1h.drop(dfinflow1h.tail(24).index,inplace=True)
        
                #rounding values to int
                result = dfinflow1h.round(0)
                result = result.convert_dtypes()
                print(result.info())
        
                result.to_csv(self.outName_30)


                print("Averaging DH ", round(time.time() - startTime,2), "s  -- done")
                print('\n')
                



"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        average = DHAverage(ADD)

        average.process_average()
