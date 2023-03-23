"""
Processes the input file and generates load data as individual files
for each country in loadData directory. Smooths results and interpolates gaps.
Input file:  Data fetched from ENTSO-E Transparency platform
Input file: input/API_token.txt
Output file(s): country_code.csv 
"""



from entsoe import EntsoePandasClient
import pandas as pd
import time
import os
import sys


class EntsoeQueryLoad:
        """
        Class for creating loads for separate areas
        """

        def __init__(self, ADD=""):
                self.ADD=ADD


                f = open(self.ADD+'input/API_token.txt', "r")
                token = f.read()

                self.client = EntsoePandasClient(api_key=token)


                self.country_codes = [
                'AT',
                'BE',	
                #'BG',
                'CH',
                #'CY',
                #'CZ',
                'DE',
                #'DE_AT_LU',
                #'DK',
                'DK_1',
                'DK_2',
                'EE',
                'ES',
                'FI',
                'FR',
                'GB',
                #'GR',
                #'HR',
                #'HU',
                #'IE',
                #'IE_SEM',
                'LT',
                #'LU',
                'LV',
                'NL',
                #'NO',
                'NO_1',
                'NO_2',
                'NO_3',
                'NO_4',
                'NO_5',
                'PL',
                #'PT',
                #'RO',
                #'SE',
                'SE_1',
                'SE_2',
                'SE_3',
                'SE_4',
                #'SI',
                #'SK'
                ]     
                self.start = pd.Timestamp('20140101', tz='Europe/Brussels') #yyyymmdd
                self.end = pd.Timestamp('20211030', tz='Europe/Brussels') #yyyymmdd

    
        def process_areas(self):
                """
                Create timeseries for each specified area 
                """
                startTime = time.time()
                print('\n')
                print('querying data -- entsoe -- query_load')

                for c in self.country_codes:
                        result = None
                        while result is None:
                                try:
                                        ts = self.client.query_load(c, start=self.start,end=self.end)
                                except:
                                        print(c, " ", round(time.time() - startTime,2), "s  -- error")
                                else:
                                        csvName = os.path.normpath(self.ADD + 'downloads/' + c +'.csv')
                                        ts.to_csv(csvName)
                                        print(c, " ", round(time.time() - startTime,2), "s  -- done")
                                        result = True

"""
# methods that return Pandas Series
client.query_day_ahead_prices(country_code, start=start,end=end)
client.query_net_position_dayahead(country_code, start=start, end=end)
client.query_load(country_code, start=start,end=end)
client.query_load_forecast(country_code, start=start,end=end)
client.query_crossborder_flows(country_code_from, country_code_to, start, end)
client.query_scheduled_exchanges(country_code_from, country_code_to, start, end, dayahead=False)
client.query_net_transfer_capacity_dayahead(country_code_from, country_code_to, start, end)
client.query_net_transfer_capacity_weekahead(country_code_from, country_code_to, start, end)
client.query_net_transfer_capacity_monthahead(country_code_from, country_code_to, start, end)
client.query_net_transfer_capacity_yearahead(country_code_from, country_code_to, start, end)
client.query_intraday_offered_capacity(country_code_from, country_code_to, start, end,implicit=True)

# methods that return Pandas DataFrames
client.query_generation_forecast(country_code, start=start,end=end)
client.query_wind_and_solar_forecast(country_code, start=start,end=end, psr_type=None)
client.query_generation(country_code, start=start,end=end, psr_type=None)
client.query_generation_per_plant(country_code, start=start,end=end, psr_type=None)
client.query_installed_generation_capacity(country_code, start=start,end=end, psr_type=None)
client.query_installed_generation_capacity_per_unit(country_code, start=start,end=end, psr_type=None)
client.query_imbalance_prices(country_code, start=start,end=end, psr_type=None)
client.query_contracted_reserve_prices(country_code, start, end, type_marketagreement_type, psr_type=None)
client.query_contracted_reserve_amount(country_code, start, end, type_marketagreement_type, psr_type=None)
client.query_unavailability_of_generation_units(country_code, start=start,end=end, docstatus=None, periodstartupdate=None, periodendupdate=None)
client.query_unavailability_of_production_units(country_code, start, end, docstatus=None, periodstartupdate=None, periodendupdate=None)
client.query_unavailability_transmission(country_code_from, country_code_to, start, end, docstatus=None, periodstartupdate=None, periodendupdate=None)
client.query_withdrawn_unavailability_of_generation_units(country_code, start, end)
client.query_import(country_code, start, end)
client.query_generation_import(country_code, start, end)
client.query_procured_balancing_capacity(country_code, start, end, process_type, type_marketagreement_type=None)
"""

"""
    Used in testing
    
"""

if __name__ == "__main__":
        ADD = "../"

        loads = EntsoeQueryLoad(ADD)

        loads.process_areas()

