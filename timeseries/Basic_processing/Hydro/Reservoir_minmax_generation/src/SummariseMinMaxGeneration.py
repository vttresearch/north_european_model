"""
Combine individual min/max generation files into one file 
Input files: countrycode.csv
Output file: summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv
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


class MinMaxSummary:
        """
        Class for summarising separate areas
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


                self.start='1982-01-01 00:00:00'
                self.end='2021-01-01 00:00:00'
                self.minvariable = 'downwardLimit'
                self.maxvariable = 'upwardLimit'
                
                self.outputName = os.path.normpath(self.ADD+'output/summary_hydro_reservoir_minmax_generation_1982_2020_1h_MWh.csv')


        def process_summary(self):
                """
                Create summary of timeseries
                """
                startTime = time.time()
                print('\n')
                print('summary PECD min max generation')

                idx = pd.MultiIndex.from_product([pd.date_range(self.start, self.end, freq='60 min'),
                                                  [self.minvariable, self.maxvariable]],
                                                 names=['time', 'boundarytype'])
                df1h = pd.DataFrame(index= idx)
                df1h.rename_axis(["",'boundarytype'], axis="rows", inplace = True)


                for c in self.country_codes:
                        print(c)
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        try:
                                indf = pd.read_csv(csvName)
                        except:
                                print("File did not exist")
                        else:
                                indf['Unnamed: 0'] = pd.to_datetime(indf['Unnamed: 0'])
                                indf.set_index(['Unnamed: 0','boundarytype'], inplace=True)
                                indf.rename_axis(["",'boundarytype'], axis="rows", inplace = True)
                                #indf.plot()
                                #pyplot.show()
                                #print(df1h.head())
                                #print(indf.head())

                                df1h = pd.merge(df1h, indf, how='left', left_index=True, right_index=True, validate='one_to_one')
                        #print(df1h.info())

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        print('\n')

                #rounding values to int
                df1h = df1h.round(0)
                df1h = df1h.convert_dtypes()
                
                df1h.to_csv(self.outputName)

        def delete_intermediates(self):
                """
                Removes input files
                """
                for c in self.country_codes:
                    csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                    if os.path.exists(csvName):
                        os.remove(csvName)


"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        summary = MinMaxSummary(ADD)

        summary.process_summary()

