import matplotlib.pyplot as plt
import gams.transfer as gt
import pandas as pd
import os

# Initialization _______________________________________________________________________________________________________

os.system('cls')

# path_to_result_gdx = os.path.normpath('C:/backbone/output/results.gdx')
path_to_output_dir = os.path.normpath('C:/backbone/output/')
path_to_result_gdx = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/RePowerEU/'
                                      '3. Model/results #4/2015-result.gdx')
# path_to_output_dir = os.path.normpath('C:/Users/jjustinas/OneDrive - Teknologian Tutkimuskeskus VTT/Desktop/RePowerEU/'
#                                       '3. Model/3. Final analysis/Figures/')

# Specifying sorted sets (to be changed by the user)

countryCodes = ['DE', 'FR', 'UK', 'ES', 'PL', 'NL', 'BE', 'SE', 'DK', 'FI', 'NO', 'LT', 'LV', 'EE']
                                                    # Sorting by population size
fuels          = pd.Series(['Nuclear',
                            'Lignite', 'Hard coal', 'Gas',                 # Fossil
                            'Biomass', 'ror', 'reservoir',                 # Flexible RE
                            'onshore', 'offshore', 'PV',                   # VRE
                            'psClosed', 'psOpen', 'batterystor',           # Storage
                            'curtailment'])    # Disbalance

fuelColorCodes = pd.Series([(0.5, 0.0, 0.6),
                            (0.0, 0.0, 0.0), (0.4, 0.4, 0.4), (0.8, 0.8, 0.8),
                            (0.0, 0.5, 0.0), (0.0, 0.0, 0.8), (0.0, 0.4, 1.0),
                            (0.0, 0.8, 1.0), (0.4, 1.0, 1.0), (1.0, 1.0, 0.0),
                            (0.5, 0.3, 0.0), (0.8, 0.5, 0.0), (1.0, 0.7, 0.3),
                            (0.8, 0.4, 1.0)])

# Data processing ______________________________________________________________________________________________________

# Reading generation and curtailment subsets

result_gdx = gt.Container(path_to_result_gdx)

gen_gn = result_gdx['r_genByFuel_gn'].records
curtailments_gn = result_gdx['r_curtailments_gn'].records         # Curtailment is present only for electricity

gen_gn.drop('grid', axis = 1, inplace = True)
curtailments_gn.drop('grid', axis = 1, inplace = True)

gen_en = gen_gn[gen_gn.node.str.contains('elec')]
gen_en = gen_en.reset_index(drop=True)

# Aggregating for countries

gen_ec = pd.DataFrame(columns=fuels, index=countryCodes)

for country in countryCodes:
    
    index_Country = gen_en.node.str.contains(country)
    
    for fuel in fuels:
        
        index_Fuel = gen_en.uni.str.contains(fuel)
        
        gen_ec.loc[country, fuel] = sum(gen_en.value[index_Country & index_Fuel])

    index_Country2 = curtailments_gn.node.str.contains(country)

    gen_ec.loc[country, 'curtailment'] = sum(curtailments_gn.value[index_Country2])

# Plot settings ________________________________________________________________________________________________________

gen_ec /= 1e6         # scaling from MWh to TWh

gen_ec.plot(kind='bar', stacked=True, color=fuelColorCodes, width=0.8)

# Gridlines

plt.grid(axis='y')

plt.minorticks_on()
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.grid(which='minor', axis='y', color=(0.9, 0.9, 0.9))

ax = plt.gca()
ax.set_axisbelow(True)

# Text elements

plt.ylabel('Generation, TWh')

plt.xticks(rotation=0)

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], ncol=2)

# Output

plt.savefig(path_to_output_dir + '\\98. P(c, f).png')
# plt.show()