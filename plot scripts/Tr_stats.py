import gams.transfer as gt
import pandas as pd
import os
import sys
from dataclasses import dataclass

# Initialization _______________________________________________________________________________________________________

os.system('cls')

path_result = os.path.normpath('C:/backbone/output/results.gdx')
path_debug  = os.path.normpath('C:/backbone/output/debug.gdx')
nodes = ['DE00', 'FR00', 'UK00', 'ES00', 'PL00', 'NL00', 'BE00', 'SE01', 'SE02', 'SE03', 'SE04', 'DKE1', 'DKW1', 'FI00',
         'NON1', 'NOM1', 'NOS0', 'LT00', 'LV00', 'EE00']

# Functions ____________________________________________________________________________________________________________

def rename_node_columns(df):
    
    df.rename(columns={'to_node': 'node_a', 'from_node': 'node_b'}, inplace=True)

    df['node_a'] = df['node_a'].str.replace('_elec', '')
    df['node_b'] = df['node_b'].str.replace('_elec', '')

    return df

def load_transfer_flows(path_result):

    result_gdx = gt.Container(path_result)

    tr_l_t  = result_gdx['r_transferLeftward_gnnft' ].records
    tr_r_t  = result_gdx['r_transferRightward_gnnft'].records

    tr_t = pd.concat([tr_l_t, tr_r_t], ignore_index=True)
        
    tr_t = tr_t[tr_t['grid'].str.contains('elec')].drop(columns=['grid', 'f'])

    tr_t = rename_node_columns(tr_t)

    tr_t['t'] = tr_t['t'].str.replace('t', '').astype(int)

    return tr_t

def load_transfer_capacities(path):

    tr_c = gt.Container(path)['p_gnn' ].records
    
    tr_c = tr_c[tr_c['grid'].str.contains('elec')]
    tr_c = tr_c[tr_c['param_gnn'].str.contains('transferCap')]
    
    tr_c = tr_c.drop(columns=['grid', 'param_gnn'])

    tr_c = rename_node_columns(tr_c)

    return tr_c

def sort_value_directions(tr, node_idx, value_col_name):

    l_mask = tr['node_a'].map(node_idx) < tr['node_b'].map(node_idx)

    tr_l = tr[ l_mask].rename(columns={'value': 'l_' + value_col_name})
    tr_r = tr[~l_mask].rename(columns={'value': 'r_' + value_col_name})

    tr_r = tr_r.rename(columns={'node_a': 'node_b', 'node_b': 'node_a'})

    merge_columns = list(tr_l.columns[:-1])
    tr = tr_l.merge(tr_r, on=merge_columns, how='outer').fillna(0)

    return tr

def sort_node_pairs(tr, node_idx):

    tr['a_order'] = tr['node_a'].map(node_idx)
    tr['b_order'] = tr['node_b'].map(node_idx)

    tr = tr.sort_values(['a_order', 'b_order']).drop(columns=['a_order', 'b_order']).reset_index(drop=True)

    return tr
# ______________________________________________________________________________________________________________________

# Flow

tr_f_t = load_transfer_flows(path_result)

node_idx = {node: i for i, node in enumerate(nodes)}

tr_f_t = sort_value_directions(tr_f_t, node_idx, 'flow')

tr_f = tr_f_t.groupby(['node_a', 'node_b'])[['l_flow', 'r_flow']].sum().reset_index()

tr_f = sort_node_pairs(tr_f, node_idx)

# Capacity

tr_c = load_transfer_capacities(path_debug)

tr_c = sort_value_directions(tr_c, node_idx, 'cap')
tr_c = sort_node_pairs      (tr_c, node_idx)

tr = tr_f.merge(tr_c, on=['node_a', 'node_b'], how='left')

# Hours at capacity

tr_f_t = tr_f_t.merge(tr_c, on=['node_a', 'node_b'], how='left')

tr_f_t['l_at_cap'] = tr_f_t['l_flow'] >= tr_f_t['l_cap']
tr_f_t['r_at_cap'] = tr_f_t['r_flow'] >= tr_f_t['r_cap']

hours_at_cap = tr_f_t.groupby(['node_a', 'node_b'])[['l_at_cap', 'r_at_cap']].sum().reset_index()
hours_at_cap = hours_at_cap.rename(columns={'l_at_cap': 'l_hours_at_cap', 'r_at_cap': 'r_hours_at_cap'})

tr = tr.merge(hours_at_cap, on=['node_a', 'node_b'], how='left')

# Utilization

tr['l_util_hours'] = tr['l_flow'] / tr['l_cap']
tr['r_util_hours'] = tr['r_flow'] / tr['r_cap']

tr['l_util_share'] = tr['l_util_hours'] / 8760
tr['r_util_share'] = tr['r_util_hours'] / 8760

# Output

mode = 'a' if os.path.exists('C:/backbone/output/Tr.xlsx') else 'w'
kwargs = {'if_sheet_exists': 'overlay'} if mode == 'a' else {}

with pd.ExcelWriter('C:/backbone/output/Tr.xlsx', engine='openpyxl', mode=mode, **kwargs) as writer:
    tr.to_excel(writer, sheet_name='Raw', index=False)