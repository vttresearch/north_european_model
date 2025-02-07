"""
Combine individual DH files into one file 
Input files: countrycode.csv
Output file: DH_2025_timeseries_summary.csv, DH_2030_timeseries_summary.csv
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


class DHSummary:
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
                'DE00',
                'DKW1',
                'DKE1',
                'EE00',
                'FI00',
                'FI_Helsinki',
                'FI_Espoo',
                'FI_Vantaa',
                'FI_Turku',
                'FI_Tampere',
                'FI_Oulu',
                'FI_Jyvaskyla',
                'LT00',
                'LV00',
                'NOS0',
                'NOM1',
                'NON1',
                'PL00',
                'SE01',
                'SE02',
                'SE03',
                'SE04'
                ]


                self.start='2010-01-01 00:00:00'
                self.end='2019-12-31 23:00:00'

                self.add = '_dheat'

                self.outName_25 = os.path.normpath(ADD+'output/DH_2025_timeseries_summary.csv')
                self.outName_30 = os.path.normpath(ADD+'output/DH_2030_timeseries_summary.csv')

        def process_summary(self):
                """
                Create timeseries for each specified area from input file
                """
                startTime = time.time()
                print('\n')
                print('summarizing DH production')
                df_25 = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
                df_30 = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))

                for c in self.country_codes:
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        temp_25 = pd.read_csv(csvName,usecols = ['Time','2025'])
                        #print(temp_25.head())
                        temp_25['Time'] = pd.to_datetime(temp_25['Time'])
                        temp_25.set_index('Time', inplace=True)
                        temp_30 = pd.read_csv(csvName,usecols = ['Time','2030'])
                        temp_30['Time'] = pd.to_datetime(temp_30['Time'])
                        temp_30.set_index('Time', inplace=True)
                        if c[0:3] == 'FI_':
                                temp_25.rename(columns={'2025': c[3:]+self.add},inplace=True)
                                temp_30.rename(columns={'2030': c[3:]+self.add},inplace=True)
                        else:
                                temp_25.rename(columns={'2025': c+self.add},inplace=True)
                                temp_30.rename(columns={'2030': c+self.add},inplace=True)
                        #indf.plot()
                        #pyplot.show()
                        
                        df_25 = pd.merge(df_25, temp_25, how='left', left_index=True, right_index=True, validate='one_to_one')
                        df_30 = pd.merge(df_30, temp_30, how='left', left_index=True, right_index=True, validate='one_to_one')
                        
                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        #print('\n')
                        
                # add FI00_others_dheat to dataframe
                df_25["FI00_others_dheat"] = df_25["FI00_dheat"]
                df_30["FI00_others_dheat"] = df_30["FI00_dheat"]
                for c in self.country_codes:
                        if c[0:3] == "FI_":
                                df_25["FI00_others_dheat"] = df_25["FI00_others_dheat"] - df_25[c[3:]+self.add]
                                df_30["FI00_others_dheat"] = df_30["FI00_others_dheat"] - df_30[c[3:]+self.add] 

                # rounding values to int
                df_25 = df_25.round(0)
                df_25 = df_25.convert_dtypes()
                df_30 = df_30.round(0)
                df_30 = df_30.convert_dtypes()

                #print(df_25.info())
                #print(df_25.head())
                #print(df_30.info())
                #print(df_30.head())

                df_25.to_csv(self.outName_25)
                df_30.to_csv(self.outName_30)

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

        summary = DHSummary(ADD)

        summary.process_summary()
