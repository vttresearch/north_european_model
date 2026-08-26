import matplotlib.pyplot as plt
import gams.transfer as gt
import numpy as np
import pandas as pd
from matplotlib import cm
from itertools import cycle
import os
from matplotlib.ticker import MultipleLocator
from dataclasses import dataclass

# Initialization _______________________________________________________________________________________________________

os.system('cls')

@dataclass
class Paths:
    result_gdx: str
    debug_gdx:  str
    output_fig: str

path = Paths(
    result_gdx = os.path.normpath('C:/backbone/output/CY1987-HY2009-results.gdx'),
    debug_gdx  = os.path.normpath('C:/backbone/output/CY1987-HY2009-debug.gdx'),
    output_fig = os.path.normpath('C:/backbone/output/S(c, g, d).png')
)

# Specifying sorted sets (to be changed by the user)

main_countries  = ['DE', 'FR', 'ES', 'BE', 'SE', 'FI', 'NO']    # Countries containing lots of long-term storage
grids           = ['hydrogenstoLT', 'psOpen', 'reservoir']

# Inputs for figure settings

country_colors   = {'DE': '#1f77b4', 'FR': '#ff7f0e', 'ES': '#2ca02c', 'BE': '#d62728', 'SE': '#9467bd',
                    'FI': '#8c564b', 'NO': '#e377c2', 'other': '#7f7f7f'}
grid_patterns    = {'hydrogenstoLT':    '',
                    'psOpen'       : '///',
                    'reservoir'    : '+++'}
grid_labels      = {'hydrogenstoLT': 'Hydrogen',
                    'psOpen'       : 'Open pumped \nhydro',
                    'reservoir'    : 'Reservoir hydro'}


month_boundaries = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
month_labels     = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Data processing ______________________________________________________________________________________________________

def process_storage_df(df, main_countries, grids):
    
    # Remove unused
    
    df = df[df['grid'].isin(grids)].copy()                      # Filter grids

    df['t'] = df['t'].astype(str).str[1:].astype(int)    # Trim notations
    df['node'] = df['node'].str[:4]

    # Aggregate

    df['d'] = df['t'] // 24 + 1                          # Time - last hours of each day
    df = df[df['t'] % 24 == 0]
    
    country = df['node'].str[:2]                                # Countries - sum 'other'
    df['country'] = country.where(country.isin(main_countries), 'other')
    df = df.groupby(['d', 'country', 'grid'], observed=True)['value'].sum().reset_index()
    
    # Reshape

    df = df.set_index(['d', 'country', 'grid'])                 # Set index
    df = df.unstack(['country', 'grid'])

    df.columns = df.columns.droplevel(0)                        # Remove first column index 'value'
    
    return df

# Read and filter

state        = gt.Container(path.result_gdx)['r_state_gnft'].records
hydro_limits = gt.Container(path.debug_gdx)['ts_node'].records

hydro_limits = hydro_limits[hydro_limits['param_gnBoundaryTypes'] != 'upwardLimit'].copy()
hydro_limits = hydro_limits[hydro_limits['f'] == 'f00'].copy()

# Process

state        = process_storage_df(state,        main_countries, grids)
hydro_limits = process_storage_df(hydro_limits, main_countries, grids)

state = state.sub(hydro_limits, fill_value=0)

# Format

state = state.reindex(columns=main_countries + ['other'], level=0)
state /= 1E6  # Scale to TWh

# Plotting _____________________________________________________________________________________________________________

plt.rcParams['font.size'] = 9

fig, ax = plt.subplots(figsize=(17/2.54, 6/2.54))
fig.tight_layout()
fig.subplots_adjust(right=0.7, left=0.08, top=0.95, bottom=0.1)

# Area plot

bottom = np.zeros(len(state))

for (country, grid) in state.columns:
    values = state[(country, grid)].fillna(0)
    if values.max() > 0:
        ax.fill_between(
            state.index, bottom, bottom + values,
            color=country_colors[country],
            hatch=grid_patterns[grid],
            alpha=0.7
        )
        bottom += values

# Legends

country_handles = []
grid_handles = []

countries = main_countries + ['other']
for country in countries:
    country_handles.append(
        plt.Rectangle((0, 0), 1, 1, 
                     facecolor=country_colors[country], 
                     alpha=1, 
                     label=country)
    )
for grid in grids:
    grid_handles.append(
        plt.Rectangle((0, 0), 1, 1, 
                     facecolor='lightgray', 
                     hatch=grid_patterns[grid],
                     edgecolor='black',
                     linewidth=0.5,
                     label=grid_labels[grid])
    )

legend1 = ax.legend(handles=country_handles, title='Countries', loc='upper left', bbox_to_anchor=(1, 1),
                   frameon=False, handlelength=1, handleheight=1)
legend2 = ax.legend(handles=grid_handles, title='Storage Types', loc='upper left', bbox_to_anchor=(1.15, 1),
                   frameon=False, handlelength=1, handleheight=1)

ax.add_artist(legend1)

ax.set_xlim(1, 365)
ax.set_ylim(0, 95)
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_minor_locator(MultipleLocator(5))

ax.set_ylabel('State of storage [TWh]')
ax.grid(True, which='major', axis='both', linestyle=':', color='black', linewidth=0.5)
ax.grid(True, which='minor', axis='y', linestyle=':', color='black', linewidth=0.5)

ax.set_xticks(month_boundaries)
ax.set_xticklabels([])

month_centers = [(month_boundaries[i] + month_boundaries[i+1]) / 2 for i in range(11)]
month_centers.append((month_boundaries[11] + 365) / 2)

for center, label in zip(month_centers, month_labels):
    ax.text(center, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02, label, ha='center', va='top')

# Output

# plt.show()
plt.savefig(path.output_fig)