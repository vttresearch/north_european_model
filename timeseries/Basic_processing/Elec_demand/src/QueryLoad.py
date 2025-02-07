"""
Processes the input file and generates load data as individual files
for each country in loadData directory. Smooths results and interpolates gaps.
Input file: time_series_60min_singleindex.csv from Open power system data
Output file(s): country_code.csv 
"""
from pandas import read_csv
from pandas import DataFrame
import pandas as pd
import time
import os
import sys
from dateutil import parser
from datetime import datetime
import numpy as np
from matplotlib import pyplot


class QueryLoad:
        """
        Class for creating loads for separate areas
        """

        def __init__(self, ADD=""):
                self.ADD=ADD
                self.file_name = os.path.normpath(ADD+'input/time_series_60min_singleindex.csv')

                self.country_codes = [
                'AT',
                'BE',	
                'CH',
                'DE',
                'DK',
                #'DK_1',
                #'DK_2',
                'EE',
                'ES',
                'FI',
                'FR',
                'GB',
                'LT',
                'LV',
                'NL',
                'NO',
                #'NO_1',
                #'NO_2',
                #'NO_3',
                #'NO_4',
                #'NO_5',
                'PL',
                'SE',
                #'SE_1',
                #'SE_2',
                #'SE_3',
                #'SE_4'
                ]

                self.add = '_load_entsoe_power_statistics'

        def process_areas(self):
                """
                Create timeseries for each specified area from input file
                """
                startTime = time.time()
                print('\n')
                print('querying and curing data -- open power system data')


                for c in self.country_codes:

                        df = pd.read_csv(self.file_name,
                                           usecols=['utc_timestamp', c+self.add])
                        #print(df.head(5))
                        #print(df.dtypes)
                        df.rename(columns={c+self.add: c}, inplace = True)
                        df['utc_timestamp'] = [parser.parse(x,ignoretz = True) for x in df['utc_timestamp']]
                        df.set_index('utc_timestamp', inplace = True)
                        df.rename_axis("", axis="rows", inplace = True)	
                        #print(df.head(5))
                        #print(df.dtypes)
                        #print(df.describe())
                        #print(df.tail(8))

                        #df.plot()
                        #pyplot.show()

                        #print('\n')
                        #print('curing load data')


                        # processing yearly data if available
                        if not df.empty:
                                
                                ###
                                # Removing single or double odd values
                                # removing outliers helps in following steps

                                # replacing zeroes with null
                                df = df.replace(0, np.nan)

                                # removing values that are below 0.5 x average
                                picks = (df / df.mean() < 0.4) 
                                df.loc[picks[c], [c]] = np.nan
                        
                                # removing values that are above 2 x average
                                picks = (df / df.mean() > 2)
                                df.loc[picks[c], [c]] = np.nan

                                # removing 1h and 2h valleys and peaks if above certain threshold
                                # calculating change / average change
                                relChange = df.diff() / df.diff().abs().mean()
                        
                                # removing 1h peaks 
                                picks = (relChange > 5) & (relChange.shift(periods=-1) < -5)
                                df.loc[picks[c], [c]] = np.nan
                                                
                                # removing 1h valleys 
                                picks = (relChange < -5) & (relChange.shift(periods=-1) > 5)
                                df.loc[picks[c], [c]] = np.nan
                        
                                # removing 2h peaks 
                                # first hour
                                picks = (relChange > 5) & (relChange.shift(periods=-2) < -5)
                                df.loc[picks[c], [c]] = np.nan
                                # second hour
                                picks = (relChange.shift(periods=1) > 5) & (relChange.shift(periods=-1) < -5)
                                df.loc[picks[c], [c]] = np.nan
                        
                                # removing 2h valleys 
                                # first hour
                                picks = (relChange < -5) & (relChange.shift(periods=-2) > 5)
                                df.loc[picks[c], [c]] = np.nan
                                # second hour
                                picks = (relChange.shift(periods=1) < -5) & (relChange.shift(periods=-1) > 5)
                                df.loc[picks[c], [c]] = np.nan

                                ###
                                # Removing longer periods of dubious data

                                # removing values that are high above the rolling average and
                                # not precided by empty value
                                dfRollingAvg = df.rolling(window=24*7, min_periods=1).mean()
                                diff = df-dfRollingAvg
                                diff[diff < 0] = 0
                                relDiff = diff / diff.mean()
                                picks = (relDiff > 8) &  (df.shift(1).sum()>0) 
                                df.loc[picks[c], [c]] = np.nan

                                # removing values that are much smaller than the rolling average 
                                dfRollingAvg = df.rolling(window=24*7, min_periods=1).mean()
                                diff = dfRollingAvg-df
                                diff[diff < 0] = 0
                                relDiff = diff / diff.mean()
                                picks = (relDiff > 8) 
                                df.loc[picks[c], [c]] = np.nan


                                ###
                                # Filling the gaps

                                # finding max 3 consecutive empty data points to be interpolated
                                indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
                                mask = df.isna().rolling(window=indexer, min_periods=1).sum() * df.isna()
                                mask = (mask.rolling(4, min_periods=1).max()) < 4* df.isna()

                                # copying df to temp, interpolate temp, pick selected values to original
                                temp = df.copy()
                                temp = temp.interpolate(method='linear', limit_area='inside')
                                df.loc[mask[c], [c]] = temp.loc[mask[c], [c]]

                                # interpolate as the last measure
                                df.interpolate(method='index', inplace=True, limit_area='inside')

                        #print(df.describe())
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        df.to_csv(csvName)
                        #df.plot()
                        #pyplot.show()

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")

                return
                
        

        
"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        loads = QueryLoad(ADD)

        loads.process_areas()


