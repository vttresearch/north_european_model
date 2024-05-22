"""
Combine individual inflow files into one file 
Input files: areas/countrycode.csv
Output file: summary_hydro_inflow_1982-2020_1h_MWh.csv
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


class InflowSummary:
        """
        Class for creating inflow summary of separate areas
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
                'SE04',
                'ITN1',
                'ITCN',
                'ITCS',
                'PT00'
                ]


                self.start='1982-01-01 00:00:00'
                self.end='2021-01-01 00:00:00'
                self.outputName = os.path.normpath(ADD+'output/summary_hydro_inflow_1982-2020_1h_MWh.csv')


        def process_summary(self):
                """
                Summarises separate areas
                """
                startTime = time.time()
                print('\n')
                print('merging PECD hydro inflows')


                dfinflow1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))


                for c in self.country_codes:
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        indf = pd.read_csv(csvName)
                        indf['Unnamed: 0'] = pd.to_datetime(indf['Unnamed: 0'])
                        indf.set_index('Unnamed: 0', inplace=True)
                        #indf.plot()
                        #pyplot.show()
                        
                        dfinflow1h = pd.merge(dfinflow1h, indf, how='left', left_index=True, right_index=True, validate='one_to_one')
                        #print(dfinflow1h.info())


                print(round(time.time() - startTime,2), "s  -- done")
                print('\n')

                #rounding values to int
                dfinflow1h = dfinflow1h.round(0)
                dfinflow1h = dfinflow1h.convert_dtypes()
                
                #print(dfinflow1h.info())

                dfinflow1h.to_csv(self.outputName)

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

        summary = InflowSummary(ADD)

        summary.process_summary()

