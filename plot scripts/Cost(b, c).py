import matplotlib.pyplot as plt
import gams.transfer as gt
import pandas as pd
import numpy as np
import os

os.system('cls')

result_folder = ('C:\\Users\\jjustinas\\OneDrive - Teknologian Tutkimuskeskus VTT\\' +
              'Desktop\\Smaller pojects\\1. NER\\2. Model\\Multiyear #2\\')

debug_gdx      = os.path.normpath(result_folder + '1987-debug.gdx')
processed_xlsx = os.path.normpath(result_folder + 'Costs.xlsx')
sheet = '1987'

bands = [(0, 50), (50, 100), (100, 200), (200, 1000), (1000, 3000)]
countries = ['DE', 'FR', 'UK', 'ES', 'PL', 'NL', 'BE', 'SE', 'DK', 'FI', 'NO', 'LT', 'LV', 'EE']

# Local functions ______________________________________________________________________________________________________

def read_nt(debug_content, gdx_variable):

    X_gnft = debug_content[gdx_variable].records
    
    X_nft = X_gnft[X_gnft.node.str.contains('elec')].reset_index(drop=True)
    X_nt  = X_nft [X_nft .f   .str.contains('f00' )].reset_index(drop=True)
    X_nt.drop(['grid', 'f'], axis=1, inplace=True)
    X_nt.node = X_nt.node.str.slice(0, 4)
    
    if X_nt.value.sum() < 0:
        X_nt.value = - X_nt.value
    
    return X_nt

def spit_into_bands(X, bands):

    for i, (floor, ceil) in enumerate(bands, 1):
        X[f'<{ceil}'] = (np.minimum(X['value'], ceil) - floor).clip(lower=0)

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
        column = f'<{bands[i-1][1]}'
        C[column] = MC[column] * D['Supplied']

    C['LL'] = 3000 * D['LL']

    return C

def aggregate_by_time_and_country(X_nt):

    X_n = X_nt.drop('t', axis=1).groupby('node').sum().reset_index()

    X_n['country'] = X_n.node.str.slice(0, 2)

    X_c = X_n.drop('node', axis=1).groupby('country').sum().reset_index()

    return X_c

def process_annual_data(debug_gdx):
    
    # Read

    debug_content  = gt.Container(debug_gdx)

    D_nt  = read_nt(debug_content, 'ts_influx')
    MC_nt = read_nt(debug_content, 'r_balance_marginalValue_gnft')
    LL_nt = read_nt(debug_content, 'r_qGen_gnft')
    LL_nt.drop(['inc_dec'], axis=1, inplace=True)

    # Process

    MC_nt = spit_into_bands               (MC_nt, bands)
    D_nt  = merge_supplied_and_lost_demand(D_nt, LL_nt)
    C_nt  = calculate_costs               (MC_nt, D_nt, bands)
    C_c   = aggregate_by_time_and_country (C_nt)

    # Format

    C_c = C_c.set_index('country').reindex(countries)

    C_c /= 1e9  # Eur to BEur

    return C_c

# ______________________________________________________________________________________________________________________

C_c = process_annual_data(debug_gdx)
print(processed_xlsx)

mode = 'a' if os.path.exists(processed_xlsx) else 'w'
if_sheet = 'overlay' if mode == 'a' else None

with pd.ExcelWriter(processed_xlsx, mode=mode, if_sheet_exists=if_sheet) as writer:
    C_c.to_excel(writer, sheet_name=sheet)

# Plotting _____________________________________________________________________________________________________________

# fig, ax1 = plt.subplots(figsize=(17/2.54, 6/2.54))

# C_c.plot(kind='bar', stacked=True, width=0.8, xlabel='', ylabel='El. costs [B€]', ax=ax1)

# labels = ["< 100", "< 1k", "< 2k", "< 3k", "LL"]
# leg = ax1.legend(labels, bbox_to_anchor=(1, 0.8), loc='upper left')
# leg.get_frame().set_edgecolor('none')

# ax1.grid(axis='y')
# plt.tight_layout()

# fig.savefig(path_outputs + '\\CY2016HY1995. C(MC threshold, c).png')
