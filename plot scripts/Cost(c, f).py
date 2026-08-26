import matplotlib.pyplot as plt
import gams.transfer as gt
import pandas as pd
import os

# Initialization _______________________________________________________________________________________________________

os.system('cls')

# path_results = os.path.normpath('C:/backbone/output/results.gdx')
path_results = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/RePowerEU/'
                                     '3. Model/results #4/2015-result.gdx')
path_outputs = os.path.normpath('C:/backbone/output/')

# Specifying sorted sets (to be changed by the user)

countryCodes = ['DE', 'FR', 'UK', 'ES', 'PL', 'NL', 'BE', 'SE', 'DK', 'FI', 'NO', 'LT', 'LV', 'EE']
                                                    # Sorting by population size
fuels = ['Nuclear',
         'Lignite', 'Hard coal',
         'Heavy oil',
         'Gas',
         'Biomass']
# Fuels found to be insignificant and thus omitted: Oil shale, Light oil, MSW

fuelColorCodes = [(0.5, 0.0, 0.6),
                  (0.0, 0.0, 0.0), (0.4, 0.4, 0.4),
                  (0.6, 0.3, 0.0),
                  (1.0, 1.0, 0.0),
                  (0.0, 0.5, 0.0)]

# Data processing ______________________________________________________________________________________________________

# Reading generation and curtailment subsets

result_gdx = gt.Container(path_results)

cost = result_gdx['r_cost_unitFuelEmissionCost_u'].records
marginalCost = result_gdx['r_balance_marginalValue_gnAverage'].records
gen_gn = result_gdx['r_gen_gn'].records

genByFuel_en = gen_gn[gen_gn.node.str.contains('elec')]
marginalCost = marginalCost[marginalCost.node.str.contains('elec')]
marginalCost = marginalCost.reset_index(drop=True)
genByFuel_en = genByFuel_en.reset_index(drop=True)

# Aggregating by country and fuels

cost_fc = pd.DataFrame(columns=fuels, index=countryCodes)
marginalCost_c = pd.DataFrame(columns=['Marginal costs'], index=countryCodes)

for country in countryCodes:
    
    # aggregating costs by summing

    iCountry = cost.unit.str[0:2].isin([country])
    
    for fuel in fuels:
        
        index_Fuel = cost.grid.str.contains(fuel)
        
        cost_fc.loc[country, fuel] = sum(cost.value[iCountry & index_Fuel])

    # aggregating marginal costs with summing normalized by node generation

    iCountry = marginalCost.node.str.contains(country)
    marginalCost_node = marginalCost.value[iCountry]

    iCountry = genByFuel_en.node.str.contains(country)
    gen_node = genByFuel_en.value[iCountry]

    marginalCost_c.loc[country] = sum(marginalCost_node*gen_node) / sum(gen_node)

# Plot settings ________________________________________________________________________________________________________

plt.rcParams['font.size'] = 9

fig, ax1 = plt.subplots(figsize=(17/2.54, 6/2.54))

# Total costs

cost_fc /= 1e3         # scaling from MEur to BEur

cost_fc.plot(kind='bar', stacked=True, color=fuelColorCodes, width=0.8, ylabel='Fuel and emission costs [B€]', ax=ax1)

plt.xticks(rotation=0)

# Marginal costs

marginalCost_c *= -1         # Making marginal costs possitive for readability

ax2 = marginalCost_c.plot(secondary_y=True, ax=ax1, style='d', color='red')
ax2.set_ylabel('\u25C6 Marginal costs [€/MWh]', color='red')
ax2.set_ylim(bottom=0)

# Gridlines

ax1.grid(axis='y')

ax1.minorticks_on()
ax1.tick_params(axis='x', which='minor', bottom=False, top=False)
ax1.grid(which='minor', axis='y', color=(0.9, 0.9, 0.9))

ax1.set_axisbelow(True)

# Legend

handles, labels = ax1.get_legend_handles_labels()

leg = ax1.legend(handles[::-1], labels[::-1], ncol=1, loc='upper left', bbox_to_anchor=(1.15, 0.8),
                 handlelength=1, handleheight=1)
leg.get_frame().set_edgecolor('none')
plt.subplots_adjust(left=0.10, right=0.75)
plt.subplots_adjust(top=0.95, bottom=0.10)

# Output

fig.savefig(path_outputs + '\\C(c, f).png')
# plt.show()
# print(cost_fc)