"""
Merges individual country files into one file 2011-2014
Input file: countrycode.csv
Output file: summary_2011-2014-1h.csv
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from matplotlib import pyplot



class MergeAreaLoads:
    """
    Class for merging loads of separate areas for years 2011-2014
    """

    def __init__(self, ADD=""):
        self.ADD=ADD
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

        self.start='2011-01-01 00:00:00'
        self.end='2014-12-31 23:00:00'


    def process_areas(self):
        startTime = time.time()
        print('\n')
        print('merging 2011-2014 load data')

        dfMerge1h = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))

        for c in self.country_codes:
            df = pd.DataFrame()
            csvInput = os.path.normpath(self.ADD+'output/'+ c +'.csv')
            df = pd.read_csv(csvInput)
            df.columns=['time', c]
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            #df.plot()
            #pyplot.show()
            dfMerge1h = pd.merge(dfMerge1h, df, how='left', left_index=True, right_index=True, validate='one_to_one')
            #print(c, " ", round(time.time() - startTime,2), "s  -- done")

        #print('\n')
        #print(dfMerge1h.info())
                
        csvOutput1h = os.path.normpath(self.ADD+'output/summary_2011-2014-1h.csv')
        dfMerge1h.to_csv(csvOutput1h)

        print("load 2011-2014 summaries ", round(time.time() - startTime,2), "s -- done")


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

        summary = MergeAreaLoads(ADD)
        
        summary.process_areas()


