import gams.transfer as gt
import pandas as pd
import os
import sys
from dataclasses import dataclass

# Variable definitions _________________________________________________________________________________________________

@dataclass
class Paths:
    result_gdx: str
    debug_gdx:  str
    output_dir: str

@dataclass
class Fuels:
    gdx_name: pd.Series
    name:     pd.Series
    color:    pd.Series

path = Paths(
    result_gdx = os.path.normpath('C:/backbone/output/results.gdx'),
    debug_gdx  = os.path.normpath('C:/backbone/output/debug.gdx'),
    output_dir = os.path.normpath('C:/backbone/output/')
    # result_gdx = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/Smaller pojects/1. NER/2. Model/Multiyear/1982-results.gdx'),
    # debug_gdx  = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/Smaller pojects/1. NER/2. Model/Multiyear/1982-debug.gdx'),
    # output_dir = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/Smaller pojects/1. NER/2. Model/Multiyear/')
)

fuel = Fuels(
    gdx_name = pd.Series(['Nuclear',
                          'Lignite', 'Hard coal', 'Oil shale', 'Gas',     # Fossil
                          'Biomass', 'ror', 'reservoir',                  # Flexible RE
                          'onshore', 'offshore', 'PV',                    # VRE
                          'psClosed', 'psOpen', 'batterystor', 'hydrogen',# Storage
                          'Elshortage', 'H2shortage', 'curtailment']),    # Disbalance
    name =    pd.Series(['Nuclear',
                          'Lignite', 'Hard coal', 'Oil shale', 'Gas',     # Fossil
                          'Biomass', 'Ror hydro', 'Reservoir hydro',      # Flexible RE
                          'Onshore wind', 'Offshore wind', 'Solar PV',    # VRE
                          'Closed PHS', 'Open PHS', 'Battery', 'hydrogen',# Storage
                          'El. shortage', 'H2 shortage', 'Curtailment',   # Disbalance
                          'Consumption']),
    color =   pd.Series([(0.5, 0.0, 0.6),
                         (0.0, 0.0, 0.0), (0.4, 0.4, 0.4), (0.6, 0.6, 0.6), (0.8, 0.8, 0.8),
                         (0.0, 0.5, 0.0), (0.0, 0.0, 0.8), (0.0, 0.4, 1.0),
                         (0.0, 0.8, 1.0), (0.4, 1.0, 1.0), (1.0, 1.0, 0.0),
                         (0.5, 0.3, 0.0), (0.8, 0.5, 0.0), (1.0, 0.7, 0.3), (0.0, 0.5, 0.5),
                         (1.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.8, 0.4, 1.0)])
)

countryCodes = ['DE', 'FR', 'UK', 'ES', 'PL', 'NL', 'BE', 'SE', 'DK', 'FI', 'NO', 'LT', 'LV', 'EE']
                                                                            # Sorted by population size
regionCodes  = ['FIN0', 'FIW0', 'FIC0', 'FIE0', 'FISW', 'FISE', 'FIUU']     # Sorted geographically from north to south

# Data processing ______________________________________________________________________________________________________

def load_data(path):

    result_gdx    = gt.Container(path.result_gdx)
    gen_gnfFt     = result_gdx['r_genByFuel_gnft'   ].records
    curtail_gnft  = result_gdx['r_curtailments_gnft'].records
    shortage_gnft = result_gdx['r_qGen_gnft'        ].records

    debug_gdx     = gt.Container(path.debug_gdx)
    influx        = debug_gdx['ts_influx'].records

    influx_elec   = influx['node'].str.contains('elec')
    influx_f00    = influx['f'] == 'f00'
    con_gnfFt     = influx[influx_elec & influx_f00].copy()
    
    return gen_gnfFt, curtail_gnft, shortage_gnft, con_gnfFt

def account_special_fuels(gen_gnfFt, curtail_gnft, shortage_gnft):
    
    curtail_gnft['uni'] = 'curtailment'
    gen_gnfFt = pd.concat([gen_gnfFt, curtail_gnft])

    if shortage_gnft is not None:
        H2shortage_gnft = shortage_gnft[shortage_gnft.node.str.contains('hydrogen')].copy()
        Elshortage_gnft = shortage_gnft[shortage_gnft.node.str.contains('elec')].copy()

        H2shortage_gnft['uni'] = 'H2shortage'
        Elshortage_gnft['uni'] = 'Elshortage'

        gen_gnfFt = pd.concat([gen_gnfFt, H2shortage_gnft, Elshortage_gnft])

    gen_gnfFt.loc[gen_gnfFt.value < 0, 'value'] = 0  # For unknown reasons there are tiny negative values
    
    return gen_gnfFt

def reindex_and_filter(gen_gnfFt, con_gnfFt):
    
    # Reindexing fuels and time
    gen_gnfFt['fuel'] = gen_gnfFt.uni.apply(lambda s: s.split('_', 1)[-1])
    gen_gnfFt.t       = gen_gnfFt.t.str.extract(r'(\d+)').astype(int)

    con_gnfFt.t       = con_gnfFt.t.str.extract(r'(\d+)').astype(int)

    if max(gen_gnfFt.t) < 168:    
            sys.exit('Error: The study period is shorter than a week. This script is not adapted to handle it.')
    
    # Dropping not relevant
    gen_gnfFt = gen_gnfFt[gen_gnfFt.node.str.contains('elec') | gen_gnfFt.uni.str.contains('H2shortage')]
    gen_gnfFt = gen_gnfFt[gen_gnfFt.f.str.contains('f00')]
    gen_nft   = gen_gnfFt.drop(['grid', 'f', 'uni'], axis=1)

    con_nt    = con_gnfFt.drop(['grid', 'f'], axis=1)

    # Reindexing nodes
    gen_nft.node = gen_nft.node.str.slice(0, 4)
    con_nt.node  = con_nt.node.str.slice(0, 4)

    return gen_nft, con_nt

def filter_fuels(gen_nft, fuel):
    
    fuelsPresent = gen_nft.fuel.unique()
    iPresent     = fuel.gdx_name.isin(fuelsPresent)

    fuel.gdx_name = fuel.gdx_name[iPresent]
    fuel.color    = fuel.color[iPresent]
    
    iPresent = pd.concat([iPresent, pd.Series([True])], ignore_index=True)
    fuel.name     = fuel.name[iPresent]

    return fuel

def aggregate_by_country(gen_nft, con_nt):
    
    gen_nft.node = gen_nft.node.str.slice(0, 2)
    con_nt.node = con_nt.node.str.slice(0, 2)

    gen_nft = gen_nft.groupby(['node', 'fuel', 't'])['value'].sum().reset_index()
    con_nt  = con_nt.groupby(['node', 't'])['value'].sum().reset_index()
    
    return gen_nft, con_nt

def aggregate_by_day(gen_nft, con_nt):

    gen_nft['d'] = (gen_nft.t - 1) // 24 + 1
    gen_nft = gen_nft.drop(columns='t')
    gen_cfd = gen_nft.groupby(['node', 'fuel', 'd'], as_index=False).sum()
    gen_cfd.value /= 24       # Scaling from MW·h per day to average MW

    con_nt['d'] = (con_nt.t - 1) // 24 + 1
    con_nt = con_nt.drop(columns='t')
    con_cd = con_nt.groupby(['node', 'd'], as_index=False).sum()
    con_cd.value /= 24

    return gen_cfd, con_cd

def pivot_and_order(gen_nft, con_nt, fuel, locationCodes, time_column):

    gen_nft = gen_nft.pivot_table(index=['node', time_column], columns='fuel', values='value')
    gen_nft = gen_nft.reindex(pd.Categorical(locationCodes), level=0)
    gen_nft = gen_nft[fuel.gdx_name]

    con_nt = con_nt.pivot_table(index=['node', time_column])
    con_nt.columns = ['consumption']
    con_nt = con_nt.reindex(pd.Categorical(locationCodes), level=0)
    
    return gen_nft, con_nt

def data_processing(path, fuel, locationCodes, t_resolution):

    gen_gnfFt, curtail_gnft, shortage_gnft, con_gnfFt = load_data(path)
    gen_gnfFt       = account_special_fuels(gen_gnfFt, curtail_gnft, shortage_gnft)

    if len(locationCodes[0]) == 4:      # if region / not country
        gen_gnfFt = gen_gnfFt[gen_gnfFt.node.str.startswith('FI')].copy()
        con_gnfFt = con_gnfFt[con_gnfFt.node.str.startswith('FI')].copy()

    gen_nft, con_nt = reindex_and_filter   (gen_gnfFt, con_gnfFt)
    fuel            = filter_fuels         (gen_nft, fuel)

    if len(locationCodes[0]) == 2:      # if country / not region
        gen_nft, con_nt = aggregate_by_country (gen_nft, con_nt)

    if t_resolution == 'd':
        gen_nft, con_nt = aggregate_by_day(gen_nft, con_nt)

    gen_nft, con_nt = pivot_and_order (gen_nft, con_nt, fuel, locationCodes, t_resolution)

    gen_nft /= 1e3   # MW →  GW
    con_nt /= -1e3   # consumption values in input data are negative

    return gen_nft, con_nt, fuel