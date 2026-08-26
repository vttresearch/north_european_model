import matplotlib.pyplot as plt
import gams.transfer as gt
import pandas as pd
import os
import sys
from matplotlib.ticker import MultipleLocator

# Initialization _______________________________________________________________________________________________________

os.system('cls')

# path_to_result_gdx = os.path.normpath('C:/backbone/output/results.gdx')
# path_to_debug_gdx  = os.path.normpath('C:/backbone/output/debug.gdx')
path_to_output_dir = os.path.normpath('C:/backbone/output/')
path_to_result_gdx = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/RePowerEU/'
                                      '3. Model/results #4/2015-result.gdx')
path_to_debug_gdx  = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/RePowerEU/'
                                      '3. Model/results #4/2015-debug.gdx')
# path_to_output_dir = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/RePowerEU/'
#                                       '3. Model/3. Final analysis/Figures/')

# Specifying sorted sets (to be changed by the user)

fuelsToPlot    = pd.Series(['Nuclear',
                            'Hard coal', 'Gas',                            # Fossil
                            'Biomass', 'ror', 'reservoir',                 # Flexible RE
                            'onshore', 'offshore', 'PV',                   # VRE
                            'psClosed', 'psOpen', 'batterystor',           # Storage
                            'Elshortage', 'H2shortage', 'curtailment'])    # Disbalance
fuelLabels     = pd.Series(['Nuclear',
                            'Hard coal', 'Gas',                            # Fossil
                            'Biomass', 'Ror hydro', 'Reservoir hydro',     # Flexible RE
                            'Onshore wind', 'Offshore wind', 'Solar PV',   # VRE
                            'Closed PHS', 'Open PHS', 'Battery',           # Storage
                            'El. shortage', 'H2 shortage', 'Curtailment',  # Disbalance
                            'Consumption', '', ''])

fuelColorCodes = pd.Series([(0.5, 0.0, 0.6),
                            (0.4, 0.4, 0.4), (0.6, 0.6, 0.6),
                            (0.0, 0.5, 0.0), (0.0, 0.0, 0.8), (0.0, 0.4, 1.0),
                            (0.0, 0.8, 1.0), (0.4, 1.0, 1.0), (1.0, 1.0, 0.0),
                            (0.5, 0.3, 0.0), (0.8, 0.5, 0.0), (1.0, 0.7, 0.3),
                            (1.0, 0.0, 0.0), (1.0, 0.5, 0.5), (0.8, 0.4, 1.0)])

t1 = 0*7*24
t2 = (31+28+31+30+31+30)*24

# Data processing ______________________________________________________________________________________________________

# Reading

result_gdx    = gt.Container(path_to_result_gdx)
gen_gnfFt     = result_gdx['r_genByFuel_gnft'   ].records
curtail_gnft  = result_gdx['r_curtailments_gnft'].records
shortage_gnft = result_gdx['r_qGen_gnft'        ].records

debug_gdx     = gt.Container(path_to_debug_gdx)
influx        = debug_gdx['ts_influx'].records

influx_elec   = influx['node'].str.contains('elec')
influx_f00    = influx['f'] == 'f00'
con_gnfFt     = influx[influx_elec & influx_f00].copy()

# Accounting curtailment as another "fuel"

curtail_gnft   ['uni'] = 'curtailment' 

gen_gnfFt = pd.concat([gen_gnfFt, curtail_gnft])

gen_gnfFt.loc[gen_gnfFt.value < 0, 'value'] = 0     # For unknown reasons there are tiny negative values

# Accounting shortages as other "fuels"

if shortage_gnft is not None:

    H2shortage_gnft = shortage_gnft[shortage_gnft.node.str.contains('hydrogen')].copy()
    Elshortage_gnft = shortage_gnft[shortage_gnft.node.str.contains('elec')].copy()

    H2shortage_gnft['uni'] = 'H2shortage'
    Elshortage_gnft['uni'] = 'Elshortage'

    gen_gnfFt = pd.concat([gen_gnfFt, H2shortage_gnft, Elshortage_gnft])

# Deriving new indexes

gen_gnfFt['fuel']    = gen_gnfFt.uni.apply(lambda s: s.split('_', 1)[-1])
gen_gnfFt.t          = gen_gnfFt.t.str.extract(r'(\d+)').astype(int)

con_gnfFt.t         = con_gnfFt.t.str.extract(r'(\d+)').astype(int)

# Dropping not relevant

gen_gnfFt = gen_gnfFt[gen_gnfFt.node.str.contains('elec') | gen_gnfFt.uni.str.contains('H2shortage')]
gen_gnfFt = gen_gnfFt[gen_gnfFt.f.str.contains('f00')]
gen_ft   = gen_gnfFt.drop(['grid', 'f', 'node', 'uni'], axis = 1)

duration  = max(gen_ft.t)

if duration < 168:
    sys.exit('Error: The study period is shorter than a week. This script is not addapted to handle it.')

con_t     = con_gnfFt.drop(['grid', 'f', 'node'], axis = 1)
con_t     = con_t[con_t['t'] <= duration]

# Aggregating node values by fuel

gen_ft = gen_ft.groupby(['fuel', 't'])['value'].sum().reset_index()
con_t  = con_t.groupby('t')['value'].sum().reset_index()

# Narrowing fuel sets down to fuels present in input data

fuelsPresent = gen_ft.fuel.unique()
iPresent     = fuelsToPlot.isin(fuelsPresent)

fuels          = fuelsToPlot   [iPresent]
fuelColorCodes = fuelColorCodes[iPresent]

# Pivoting and ordering

gen_ft = gen_ft.pivot_table(index='t', columns='fuel', values='value')
gen_ft = gen_ft[fuels]

con_t = con_t.pivot_table(index='t')
con_t.columns = ['consumption']

# Plot settings ________________________________________________________________________________________________________

gen_ft /= 1e3         # scaling from MW to GW
con_t /= -1e3         # consumption values in input data are negative

# Figure

plt.rcParams['font.size'] = 9
fig, axes = plt.subplots(1, 2, figsize=(17/2.54, 8/2.54))

week_starts = [t1, t2]

for i, start in enumerate(week_starts):
    
    ax = axes[i]

    gen_week = gen_ft.loc[start:start + 168]
    con_week = con_t.loc[start:start + 168]

    # Plot stacked area for generation by fuel
    gen_week.plot.area(
        stacked=True,
        ax=ax,
        legend=False,
        xlim=[start, start + 168],
        color=fuelColorCodes,
        linewidth=0
    )

    # Plot total consumption
    ax.plot(con_week, color='black', label='El. demand', linewidth=1)

    # Axes and grid
    last_hour_of_day = [start + x for x in [24, 48, 72, 96, 120, 144, 168]]
    ax.set_xticks(last_hour_of_day)
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7])
    if i == 0:
        ax.set_xlabel('January days')
    else:
        ax.set_xlabel('July days')

    ax.set_ylim(0, 800)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(50))

    ax.grid(True, which='major', axis='y', linestyle='--', color='black', linewidth=0.5)
    ax.grid(True, which='minor', axis='y', linestyle=':', color='black', linewidth=0.5)
    ax.grid(True, which='major', axis='x', linestyle=':', color='black', linewidth=0.5)

    # move y-axis of second subplot to the right
    if i == 1:
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

# Legend and subtitle

handles, labels = ax.get_legend_handles_labels()

# Add two empty patches to shift content down
from matplotlib.patches import Patch
empty_patch_1 = Patch(color='none', label='')
empty_patch_2 = Patch(color='none', label='')

handles = handles + [empty_patch_1, empty_patch_2]
# labels = labels + ['', '']

leg = fig.legend(handles[::-1], fuelLabels[::-1],
                 loc='lower center', ncol=6, facecolor='none', handlelength=1, handleheight=1, columnspacing=0.6)
leg.get_frame().set_edgecolor('none')

plt.suptitle('Generation [GW]', fontsize=9)
plt.subplots_adjust(left=0.06, right=0.94)
plt.subplots_adjust(top=0.92, bottom=0.38)
plt.subplots_adjust(wspace=0, hspace=0)

# Output
plt.savefig(path_to_output_dir + '\\P(f, h).png')