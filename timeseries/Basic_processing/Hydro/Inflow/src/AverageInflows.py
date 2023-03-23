"""
Create average year of inflows from available years 
Input files: countrycode.csv
Output file: summary_hydro_average_year_1982-2020_1h_MWh.csv
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

class AverageInflow:
        """
        Class for creating average year of inflows for areas that have inflows
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
                'SE04'
                ]


                self.outputName = os.path.normpath(ADD+'output/summary_hydro_average_year_1982-2020_1h_MWh.csv')


        def process_average(self):
                """
                Processing average from years that are available
                """
                startTime = time.time()
                print('\n')
                print('creating average years of PECD hydro inflows')


                dfinflow1h = pd.DataFrame()


                for c in self.country_codes:
                        avg_inflow = pd.DataFrame()
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        indf = pd.read_csv(csvName)
                        indf['Date'] = pd.to_datetime(indf['Unnamed: 0'])
                        del indf['Unnamed: 0']
                        #when processing inflow data the 30th date was copied to 31st for leap years,
                        #therefore those additional 24 hours are removed from leap years before further processing
                        indf['is_leap'] = indf['Date'].dt.year.apply(lambda e: calendar.isleap(e))
                        indf.drop(indf.loc[(indf['is_leap'])&(indf['Date'].dt.month == 12)&(indf['Date'].dt.day == 31)].index, inplace=True)
                        indf.reset_index(drop = True, inplace=True)
                        indf['group'] = indf.index % 8760
                        del indf['Date']
                        del indf['is_leap']
                        #group data in 8760 rows and take average of each row
                        if (not indf[c+'_reservoir'].dropna().empty) and (indf[c+'_reservoir'].sum()!=0):
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_reservoir'].quantile(0.1)
                                dfinflow1h['10_'+c+'_reservoir'] = avg_inflow[c+'_reservoir']
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_reservoir'].quantile(0.5)
                                dfinflow1h['50_'+c+'_reservoir'] = avg_inflow[c+'_reservoir']
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_reservoir'].quantile(0.9)
                                dfinflow1h['90_'+c+'_reservoir'] = avg_inflow[c+'_reservoir']
                                #dfinflow1h[['10_'+c+'_reservoir','50_'+c+'_reservoir','90_'+c+'_reservoir']].plot()
                                #pyplot.show()
                        if (not indf[c+'_psOpen'].dropna().empty) and (indf[c+'_psOpen'].sum()!=0):
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_psOpen'].quantile(0.1)
                                dfinflow1h['10_'+c+'_psOpen'] = avg_inflow[c+'_psOpen']
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_psOpen'].quantile(0.5)
                                dfinflow1h['50_'+c+'_psOpen'] = avg_inflow[c+'_psOpen']
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_psOpen'].quantile(0.9)
                                dfinflow1h['90_'+c+'_psOpen'] = avg_inflow[c+'_psOpen']
                        if (not indf[c+'_ror'].dropna().empty) and (indf[c+'_ror'].sum()!=0):
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_ror'].quantile(0.1)
                                dfinflow1h['10_'+c+'_ror'] = avg_inflow[c+'_ror']
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_ror'].quantile(0.5)
                                dfinflow1h['50_'+c+'_ror'] = avg_inflow[c+'_ror']
                                avg_inflow = indf.groupby([indf.group], as_index = False)[c+'_ror'].quantile(0.9)
                                dfinflow1h['90_'+c+'_ror'] = avg_inflow[c+'_ror']                


                #rounding values to int
                result = dfinflow1h.round(0)
                result = result.convert_dtypes()

                #print(result.info())

                result.to_csv(self.outputName)
                print(round(time.time() - startTime,2), "s  -- done")
                print('\n')


"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        summary = AverageInflow(ADD)

        summary.process_average()

