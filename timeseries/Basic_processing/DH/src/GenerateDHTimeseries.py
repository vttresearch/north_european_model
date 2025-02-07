"""
Processing district heating time series
Input files:DH summary.xlsx, Temperature.csv
Output files: countrycode.csv
"""

from pandas import read_csv
from pandas import read_excel
from pandas import DataFrame
import pandas as pd
import time
import os
import sys
from dateutil import parser
from datetime import date, datetime
import numpy as np
from matplotlib import pyplot


class GenerateDH:
        """
        Class for creating timeseries for separate areas
        """

        def __init__(self, ADD=""):
                """
                Initialize variables
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
                self.start = '2010-01-01 00:00:00'
                self.end = '2019-12-31 23:00:00'

                self.inputName = os.path.normpath(ADD+'input/DH summary.xlsx')
                self.tempName = os.path.normpath(ADD+'input/Temperature.csv')

        def process_areas(self):
                """
                Main functionality
                """
                startTime = time.time()
                print('\n')
                print('Creating DH timeseries')

                dfc = pd.DataFrame()
                production_df = read_excel(self.inputName,sheet_name='Production')
                production_df.columns = production_df.columns.astype(str)

                for c in self.country_codes:
                        #print(c)
                        csvName = os.path.normpath(self.ADD+'output/'+ c +'.csv')
                        #Select production values to be processed
                        pdf = production_df.loc[production_df['Area'] == c].copy()
                        pdf.set_index("Parameter", drop=True, inplace=True)
                        #print(pdf.head())
                        #print(pdf.info())
                        #Process temperatures 
                        indf = pd.read_csv(self.tempName)
                        indf['Time'] = pd.to_datetime(indf['Time'])
                        dflist = []
                        indf['Heating'] = 17 - indf[c[0:2]]
                        indf['Heating'] = indf['Heating'].clip(lower=0).fillna(0) #negative values to 0
                        rate_30_25 = pdf.at["Space heating",'2030']/pdf.at["Space heating",'2025'] #rate for transforming 2025 results to 2030
                        for year in range(pd.to_datetime(self.start).year,(pd.to_datetime(self.end).year)+1):
                                #print(year)
                                x = indf[indf['Time'].dt.year==year].copy()
                                hours = len(x)
                                sumtemp = x['Heating'].sum()
                                sh_25 = pdf.at["Space heating",str(year)]   #space heating sum for year
                                water = pdf.at["Water heating",str(year)]/hours #water heating for each hour in year
                                x['2025'] = (x['Heating']/sumtemp * sh_25 + water)*1000    #from GWh to MWh
                                x['2030'] = (x['Heating']/sumtemp * sh_25 * rate_30_25 + water)*1000 #from GWh to MWh
                                del x['Heating']
                                del x[c[0:2]]
                                x.set_index('Time',drop=True, inplace=True)
                                y = x.loc[:,['2025','2030']].copy()
                                #print(x.head())
                                dflist.append(y) #append one year at a time to list
                        dfc=pd.concat(dflist,axis=0) #combine all years

                        #print(dfc.info())
                        #print(dfc.head(10))
                        
                        dfc.to_csv(csvName)

                        #dfc.plot()
                        #pyplot.show()

                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                        #print('\n')

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        DH = GenerateDH(ADD)

        DH.process_areas()
