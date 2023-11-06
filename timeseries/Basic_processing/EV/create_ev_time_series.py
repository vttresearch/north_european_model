#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this script requires PI_calculations.xlsm and region_coefficients.csv in the same folder
# it also requires pandas and openpyxl that can be installed with pip

import pandas as pd
import itertools


# In[2]:

# scenario and year of EV capacity, e.g. "Distributed Energy", 2030
scenario = "Distributed Energy"
year = 2040

# years between 1982-2020 (inclusive): 39
years_qty = 39
# qty of leap years between 1982-2020: 10
    # in 84, 88, 92, 96, 00, 04, 08, 12, 16, 20
leap_years_qty = 10
# hourly timesteps for every year plus one extra day per leap year
timesteps_qty = 24*(365*years_qty + leap_years_qty)
# the Backbone timestep timeseries (of form 't000123')
timesteps = list(map(lambda x: 't' + f'{x}'.zfill(6), range(1,timesteps_qty + 1)))



print('Reading PI_calculations.xlsx...')
# unedited ev data from PI_calculations.xlsm for full year
ev_data_unedited = pd.read_excel('PI_calculations.xlsx',
                        sheet_name='perus',
                        usecols='D,S,U,W',
                        names=['finnish_time','ts_influx','ts_cf','ts_node'],
                        skiprows=11,
                        nrows=8760)
#print('Reading region_coefficients.csv...')
#coefficient_data_raw = pd.read_csv('region_coefficients.csv').fillna(0).query('coefficient != 0')


print('Reading EV fleet by region...')
coefficient_data_raw = pd.read_excel('region_coefficients.xlsx', sheet_name='Capacity')
coefficient_data_raw = coefficient_data_raw[(coefficient_data_raw["Scenario"] == scenario) & 
                                            (coefficient_data_raw["Year"] == year)]
# Rename the Node column
coefficient_data_raw = coefficient_data_raw.rename(columns={'Node': 'region'})

regions_with_coefficients = list(coefficient_data_raw['region'])


# names of the country nodes divided into timezones
country_nodes_utc = {
    0:['UK00'],
    1:['SE01', 'SE02', 'SE03', 'SE04',
       'NOS0', 'NOM1', 'NON1','DKE1',
       'DKW1', 'NL00', 'BE00','FR00',
       'DE00', 'PL00'],
    2:['FI00', 'EE00', 'LT00', 'LV00']
}
print(f'Preparing time series for {len(regions_with_coefficients)} bidding areas: {regions_with_coefficients}.')
# filter country nodes that have coefficients
allnodes = list(itertools.chain.from_iterable(country_nodes_utc.values()))
nodes_with_coefficients = []
for region in regions_with_coefficients:
    if region in allnodes:
        nodes_with_coefficients.append(region)


coefficient_data = coefficient_data_raw.copy()
coefficient_data['region'] = nodes_with_coefficients


def coefficient(node_name:str):
    return coefficient_data[coefficient_data['region']==node_name]['coefficient'].values[0]


# unedited data starts from Saturday, the Nordic time step count from Friday (1982-01-01 00:00)
#  --> we copy the friday of the following week (starting at 24*6 hours) to the beginning of the dataframe
# we also only take only the first 364 days to have 52 full weeks
# we assume same kind of behaviour for every country and just shift the time count to utc time
# the first hour is skipped for UTC+1 and two hours for UTC+2 countries
    # one or two extra hours are taken for UTC+1 and UTC+2 countries from day 365

ev_data_shifted_utc = pd.concat([ev_data_unedited[24*6:24*7], ev_data_unedited])[0:24*364].reset_index(drop=True)
ev_data_shifted_utc_plus_1 = pd.concat([ev_data_unedited[24*6:24*7], ev_data_unedited])[1:24*364+1].reset_index(drop=True)
ev_data_shifted_utc_plus_2 = pd.concat([ev_data_unedited[24*6:24*7], ev_data_unedited])[2:24*364+2].reset_index(drop=True)


# In[8]:


# this will contain the extended ts_influx, ts_cf and ts_node time series for all time zones
# the index will be the time zone (utc + 0, utc + 1, utc + 2)
ev_data_all_years_all_time_zones = []

for ev_data in [ev_data_shifted_utc, ev_data_shifted_utc_plus_1, ev_data_shifted_utc_plus_2]:
    ev_data_all_years = ev_data
    counter = 0
    for i in range(1, years_qty):
        counter += 1
        # same Finnish data is used for every year
        ev_data_all_years = pd.concat([ev_data_all_years, ev_data])
        if counter % 7 == 0:
            # add a missing week after every seven years
            # a week gets added five times at counter values 7, 14, 21, 28, 35
            # four days (corresponding to counter values 36, 37, 38, 39) are added later
            ev_data_all_years = pd.concat([ev_data_all_years, ev_data[0:24*7]])
    # add missing days: leap_years_qty + 4 days missing from above loop
    ev_data_all_years = pd.concat([ev_data_all_years, ev_data[0:24*(leap_years_qty + 4)]])
    ev_data_all_years = ev_data_all_years.reset_index(drop=True)
    ev_data_all_years_all_time_zones.append(ev_data_all_years)


# In[9]:


# UTC+0 (Winter) and UTC+1 (DST): UK
    # same time series as in ev_data_all_years
# UTC+1 (Winter) and UTC+2 (DST): SE, NO, DK, NL, BE, FR, DE, PL
# UTC+2 (Winter) and UTC+3 (DST): FI, Estonia (EE), Latvia (LT), Lithuania (LV)

# timezone is either 0 (UTC), 1 (UTC+1) or 2 (UTC+2)
def create_final_dataframes(df:pd.DataFrame, nodes_utc:list, ts_name:str):
    # links time series data for each country (or bidding area) node in each timezone
    for timezone in range(0,3):
        for node in nodes_utc[timezone]:
            if node in nodes_with_coefficients:
                if ts_name == 'ts_unit':
                    colname = "U_" + node + "_EVsmartcha"
                    df[colname] = ev_data_all_years_all_time_zones[timezone]['ts_cf']
                else:
                    colname = node + "_ev"
                    df[colname] = coefficient(node) * ev_data_all_years_all_time_zones[timezone][ts_name]
    
    # add t and f columns (included in all time series)
    df.insert(0,'t', timesteps)
    df.insert(0, 'f', 'f00')

    # add columns specific to each time series
    # ts_cf columns: flow, f, t, nodes
    if ts_name == 'ts_unit': # ts_cf
        df.insert(0, 'param_unit', 'availability') # flow, ev_connected
    # ts_node columns: grid, param_gnboundarytypes, f, t, nodes
    if ts_name == 'ts_node':
        df.insert(0, 'param_gnboundarytypes', 'upwardLimit')
    # ts_influx columns: grid, f, t, nodes
    if ts_name in ['ts_influx', 'ts_node']:
        df.insert(0, 'grid', 'all')
        
            
        
ts_influx, ts_unit, ts_node = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

# use create the final dataframes
create_final_dataframes(ts_influx, country_nodes_utc, 'ts_influx')
create_final_dataframes(ts_unit, country_nodes_utc, 'ts_unit') 
create_final_dataframes(ts_node, country_nodes_utc, 'ts_node')


# create csv files
print('Creating ts_influx.csv, ts_unit.csv and ts_node.csv...')
ts_influx.to_csv('bb_ts_influx_ev.csv', index=False, float_format="%.4f")
ts_unit.to_csv('bb_ts_unit_ev.csv', index=False, float_format="%.4f")
ts_node.to_csv('bb_ts_node_ev.csv', index=False, float_format="%.4f")

print('Done.')

