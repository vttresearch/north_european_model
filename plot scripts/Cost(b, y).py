import matplotlib.pyplot as plt
import gams.transfer as gt
import pandas as pd
import numpy as np
import os

os.system('cls')

# path_debug_foler = 'C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/RePowerEU/3. Model/results #4/'
path_debug_foler = 'C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/Smaller pojects/1. NER/2. Model/Multiyear/'
# path_results = os.path.normpath('C:/backbone/output/results.gdx')
path_outputs = os.path.normpath('C:/backbone/output/')

# years = ['2016', '2010', '1987', '1995', '2002', '2009', 'CY2016-HY1995', 'CY2010-HY2002', 'CY1987-HY2009']
years = [str(y) for y in range(1982, 2017)]

bands = [(0, 50), (50, 100), (100, 200), (200, 1000), (1000, 3000)]
countries = ['DE', 'FR', 'UK', 'ES', 'PL', 'NL', 'BE', 'SE', 'DK', 'FI', 'NO', 'LT', 'LV', 'EE']

# Local functions ______________________________________________________________________________________________________

def read_nt(gdx_file, gdx_variable):

    X_gnft = gdx_file[gdx_variable].records
    
    X_nft = X_gnft[X_gnft.node.str.contains('elec')].reset_index(drop=True)
    X_nt  = X_nft [X_nft .f   .str.contains('f00' )].reset_index(drop=True)
    X_nt.drop(['grid', 'f'], axis=1, inplace=True)
    X_nt.node = X_nt.node.str.slice(0, 4)
    
    if X_nt.value.sum() < 0:
        X_nt.value = - X_nt.value
    
    return X_nt

def spit_into_bands(X, bands):

    for i, (floor, ceil) in enumerate(bands, 1):
        X[f'band_{i}'] = (np.minimum(X['value'], ceil) - floor).clip(lower=0)

    X.drop(['value'], axis=1, inplace=True)
    
    return X

def merge_supplied_and_lost_demand(D, LL):

    D = D.rename(columns={'value': 'Demand'})
    LL = LL.rename(columns={'value': 'LL'})

    D = D.merge(LL[['node', 't', 'LL']], on=['node', 't'],  how='left')
    D['LL'] = D['LL'].fillna(0)
    D['Supplied'] = D['Demand'] - D['LL']

    return D

def calculate_costs(MC, D, bands):

    C = MC.copy()

    for i in range(1, len(bands) + 1):
        C[f'band_{i}'] = MC[f'band_{i}'] * D['Supplied']

    C['band_LL'] = 3000 * D['LL']

    return C

def process_annual_data(debug_gdx):
    
    # Read

    debug_gdx  = gt.Container(path_debug)

    D_nt  = read_nt(debug_gdx, 'ts_influx')
    MC_nt = read_nt(debug_gdx, 'r_balance_marginalValue_gnft')
    
    if debug_gdx['r_qGen_gnft'].records is None:
        LL_nt = pd.DataFrame(columns=['node', 't', 'value'])
    else:
        LL_nt = read_nt(debug_gdx, 'r_qGen_gnft')
        LL_nt.drop(['inc_dec'], axis=1, inplace=True)

    # Process

    MC_nt = spit_into_bands               (MC_nt, bands)
    D_nt  = merge_supplied_and_lost_demand(D_nt, LL_nt)
    C_nt  = calculate_costs               (MC_nt, D_nt, bands)
    C_c   = C_nt.drop(['node', 't'], axis=1).sum()

    C_c /= 1e9  # Eur to BEur

    return C_c

# Process ______________________________________________________________________________________________________________

C_y = []

for y in years:
    
    print(f'Processing {y}...')
    
    path_debug = os.path.normpath(path_debug_foler + y + '-debug.gdx')
    C = process_annual_data(path_debug)
    C_y.append(C)

C_y = pd.DataFrame(C_y, index=years)
print(C_y)
print(type(C_y))
