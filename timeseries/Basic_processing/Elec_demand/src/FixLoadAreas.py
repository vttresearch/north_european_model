"""
As input data for years 2011-2014 did not have areas for DK, NO, and SE,
only sums, the distribution rates from years 2015-2020 were used to calculate
the area values.
Input files: rates_for_areas.csv,rates_for_areas_leap_year.csv, country_code.csv (sums)
Output file: country_code.csv
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
import calendar

class FixLoadAreas:
        """
        Class for fixing loads for separate areas
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
                    2011,
                    2012,
                    2013,
                    2014
                    }



                self.filename = os.path.normpath(ADD+'output/rates_for_areas.csv')
                self.in_rates = pd.read_csv(self.filename)
                self.leap_filename = os.path.normpath(ADD+'output/rates_for_areas_leap_year.csv')
                self.in_leap_rates = pd.read_csv(self.leap_filename)
                #extend for number of years
                self.end ="2015-01-01"

        def process_fix(self):
                """
                Processes areas that need fixing
                """
                startTime = time.time()
                print('\n')
                print('open power system data -- fixing area loads')

                all_res = []

                for y in self.years:
                        if calendar.isleap(y):
                                all_res.append(self.in_leap_rates)
                        else:
                                all_res.append(self.in_rates)
                          
                rates = pd.concat(all_res)
                #print(rates.describe())
                for c in self.country_codes:
                        file_name = os.path.normpath(self.ADD+'output/'+ c[0:2] +'.csv')
                        df = pd.read_csv(file_name)
                        df['Unnamed: 0'] =  pd.to_datetime(df['Unnamed: 0'])
                        df = df[df['Unnamed: 0'] < self.end]
                        c_name = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        df[c] = df[c[0:2]].mul(rates[c].values)
                        df.set_index('Unnamed: 0', inplace = True)
                        df.rename_axis("", axis="rows", inplace = True)
                        del df[c[0:2]]
                        #print(df.head(5))
                        df.to_csv(c_name)

                print("areas calculated", round(time.time() - startTime,2), "s  -- done")

        def delete_intermediates(self):
                """
                Removes input files
                """
                if os.path.exists(self.filename):
                        os.remove(self.filename)
                if os.path.exists(self.leap_filename):
                        os.remove(self.leap_filename)
                for c in self.country_codes:
                        file_name = os.path.normpath(self.ADD+'output/'+ c[0:2] +'.csv')
                        if os.path.exists(file_name):
                                os.remove(file_name)


"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        fixes = FixLoadAreas(ADD)
        
        fixes.process_fix()

