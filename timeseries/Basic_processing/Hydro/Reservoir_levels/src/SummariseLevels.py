"""
Combine individual reservoir level files into one file 
Input files: countrycode.csv
Output file: summary_historical_hydro_reservoir_levels_1h_MWh.csv
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

class LevelsSummary:
        """
        Class for creating historical reservoir levels summary of separate areas
        """

        def __init__(self, ADD=""):
                """
                Initialize variables
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
                self.fileName = os.path.normpath(ADD+'output/summary_historical_hydro_reservoir_levels_1h_MWh.csv')


        def process_summary(self):
                """
                Summarises separate areas
                """
                startTime = time.time()
                print('\n')
                print('summary ensoe-e transparency historical reservoir levels')

                df1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))

                for c in self.country_codes:
                        print(c)
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        try:
                                indf = pd.read_csv(csvName)
                        except:
                                print("File did not exist")
                        else:
                                indf['Unnamed: 0'] = pd.to_datetime(indf['Unnamed: 0'])
                                indf.set_index('Unnamed: 0', inplace=True)

                                df1h = pd.merge(df1h, indf, how='left', left_index=True, right_index=True, validate='one_to_one')
                                if c[:2] == 'NO':
                                        df1h.rename(columns={c: c+'_psOpen'}, inplace=True)
                                else:
                                        df1h.rename(columns={c: c+'_reservoir'}, inplace=True)
                        #print(df1h.info())

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        print('\n')

                #rounding values to int
                df1h = df1h.round(0)
                df1h = df1h.convert_dtypes()
                
                df1h.to_csv(self.fileName)

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

        summary = LevelsSummary(ADD)

        summary.process_summary()
