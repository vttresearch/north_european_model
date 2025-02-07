
"""
Load summary files from 2011-2014 and 2015-2020 are combined to one file.
As there is a jump in values for CH from one dataset to another, the exarlier values
are "corrected" to be more inline with later ones. However, file without the fix is also generated.
Input files: summary_2011-2014-1h.csv, summary_2015-2020-1h.csv, Temperature.xlsx
Output file: summary_2011-2020-1h.csv
"""


import pandas as pd
from pandas import read_excel
import numpy as np
import time
import os
import sys
import datetime
from matplotlib import pyplot


class CombineSummaries:
    """
    Class for merging loads of separate areas for years 2011-2014
    """

    def __init__(self, ADD="", scenario = "", year = 0):
        
        self.ADD=ADD
        self.scenario = scenario
        self.year = year

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
        self.end='2019-12-31 23:00:00'              #temperatures are available until this date
        self.middle = '2015-01-01 00:00:00'
        #self.countrycodefile = os.path.normpath(self.ADD+'input/countrycodes.csv')

        #region codes
        #country_codes1 = pd.read_csv(self.countrycodefile)

        #files to be combined
        self.csvInput = os.path.normpath(self.ADD+'output/summary_2011-2014-1h.csv')
        self.laterInput = os.path.normpath(self.ADD+'output/summary_2015-2020-1h.csv')
        
        #Process temperatures
        temp_name = os.path.normpath(ADD+'input/Temperature.csv')
        indf = pd.read_csv(temp_name)
        indf['Time'] = pd.to_datetime(indf['Time'])
        #between start and end
        indf = indf[indf["Time"] <= pd.to_datetime(self.end)]
        indf = indf[indf["Time"] >= pd.to_datetime(self.start)]
        indf.set_index('Time', inplace=True)
        self.temps = indf.copy()
        #print(indf.info())
        
        #Importing constants file for temperature-specific adjustments
        #from excel file if scenario is defined
        if self.scenario != "":
        #    constant_name = os.path.normpath(ADD+'input/constants.csv')
        #    self.constants = pd.read_csv(constant_name)
        #    self.constants.set_index("Constant",inplace=True)
        #else:
            constant_name = os.path.normpath(ADD+'input/demand_coefficients.xlsx')
            a = pd.read_excel(constant_name, "constants")
            a = a[(a["Scenario"] == self.scenario) & (a["Year"] == self.year) ]  
            a.drop(["Scenario", "Year"], axis=1, inplace=True)
            a.set_index("Constant", inplace=True)
            self.constants = a


        #print(self.constants.head())
        
    def update_with_temperature(self, df):
        final = pd.DataFrame()
        for c in self.country_codes:
            df["C_1"] = self.constants.at["C_1", c] * self.temps[c[0:2]] + self.constants.at["C_0", c]
            final[c] = df[c] + df["C_1"]  
            del df["C_1"]
        return final

    def process_summary(self):
        startTime = time.time()
        print('\n')
        print('combining summaries 2011-2020')

        df = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))
        final = pd.DataFrame(index = pd.date_range(self.start, self.end, freq='60 min'))

        early = pd.read_csv(self.csvInput)
        early.set_index('Unnamed: 0', inplace=True)
        #print(early.info(5))

        olater = pd.read_csv(self.laterInput)
        olater.set_index('Unnamed: 0', inplace=True)
        oolater = olater[olater.index <= self.end]
        later = oolater[oolater.index >= self.middle]

        #print(later.info())

        #LT values missing from the beginning
        indices = np.where(later['LT'].isnull())[0]
        for i in indices:
            #print(later.at[later.index[i],'LT'])
            later.at[later.index[i],'LT'] = later.at[later.index[i+24],'LT']
            #print(later.at[later.index[i],'LT'])

        #print(later.info())

        df = pd.concat([early,later],join='inner')
        df.rename_axis("", axis="rows", inplace = True)	

        #print('\n')
        #print(df.info())
        """
        for c in country_codes:
            print('\n')
            print(c)    
            df[c].plot()
            pyplot.show()
        """
                
        #csvOutput1h_notfixed = os.path.normpath('../output/summary_2011-2020-1h_without_CH_fix.csv')
        #df.to_csv(csvOutput1h_notfixed)

        #CH values differ when compared to early and later values
        c = 'CH'
        #print(early[c].mean())
        #print(later[c].mean())

        fix_rate = later[c].mean() / early[c].mean()
        #print(fix_rate)
        #print('\n')

        early.loc[:,c] = early.loc[:,c]*fix_rate
        #print(early[c].mean())
        #print(later[c].mean())

              
        final = pd.concat([early,later],join='inner')
        final.rename_axis("", axis="rows", inplace = True)

        """
        for c in country_codes:
            print('\n')
            print(c)    
            final[c].plot()
            pyplot.show()
        """

        # Update load with given demand coefficients if scenario is defined
        if self.scenario != "":
            result = self.update_with_temperature(final)
        else:
            result = final
        
        # change values to ints
        resultInt = result.round(0)
        resultInt = resultInt.convert_dtypes()
        #print(resultInt.info())
        
        csvOutput1h = os.path.normpath(self.ADD+'output/summary_load_2011-2020-1h.csv')
        resultInt.to_csv(csvOutput1h)


        print("load summaries ", round(time.time() - startTime,2), "s  -- done")


    def delete_intermediates(self):
        """
        Removes input files
        """
        if os.path.exists(self.csvInput):
            os.remove(self.csvInput)
        if os.path.exists(self.laterInput):
            os.remove(self.laterInput)
"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        summary = CombineSummaries(ADD)        
        summary.process_summary()
