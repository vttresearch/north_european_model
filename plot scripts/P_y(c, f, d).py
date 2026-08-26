import matplotlib.pyplot as plt
import os
from plot_utils import path, fuel, countryCodes, data_processing

os.system('cls')

month_labels     = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_boundaries = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]

gen_cfd, con_cd, fuel = data_processing(path, fuel, countryCodes, 'd')

# Plot settings ________________________________________________________________________________________________________

plt.rcParams['font.size'] = 9

# Figure

fig, axs = plt.subplots(7, 2, figsize=(17/2.54, 23/2.54))

for i, country in enumerate(countryCodes):
    
    ax = axs[i%7, i//7]

    gen = gen_cfd.loc[country]
    gen.plot.area(stacked=True, ax=ax, legend=False, xlim=[0, 364], color=fuel.color, linewidth=0)

    con = con_cd.loc[country]
    con.plot(ax=ax, legend=False, xlim=[0, 364], color='black', linewidth=0.5)

    ax.text(0.5, 0.85, country, transform=ax.transAxes)

# Axes

for row in axs:

    for columnNr, ax in enumerate(row):
        
        # Gridlines

        ax.grid(True, which='major', axis='both', linestyle=':', color='black', linewidth=0.5)
        ax.grid(True, which='minor', axis='y', linestyle=':', color='black', linewidth=0.5)

        # OX: ticks by days

        ax.set_xticks(month_boundaries)
        ax.set_xticklabels([])

        month_centers = [(month_boundaries[i] + month_boundaries[i+1]) / 2 for i in range(11)]
        month_centers.append((month_boundaries[11] + 365) / 2)

        for center, label in zip(month_centers, month_labels):
            ax.text(center, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1, label, ha='center', va='top')
        
        ax.set_xlabel('')

        # OY: uniform tick number (not always work) without excesive labels

        if columnNr == 1:
            ax.yaxis.tick_right()

        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_ylim(0, ax.get_yticks()[-1])

        tick_labels = ax.get_yticklabels()

        tick_labels[0].set_visible(False)
        tick_labels[1].set_visible(False)
        tick_labels[3].set_visible(False)

# Legend and subtitle

handles, labels = axs[0, 0].get_legend_handles_labels()
leg = fig.legend(handles[::-1], fuel.name[::-1],
                 loc='lower center', ncol=6, handlelength=1, handleheight=1, columnspacing=0.6)
leg.get_frame().set_edgecolor('none')

plt.suptitle('Generation [GW]', fontsize=9)
plt.subplots_adjust(top=0.96, bottom=0.13)
plt.subplots_adjust(left=0.05, right=0.95)
plt.subplots_adjust(wspace=0, hspace=0)

# Output

plt.savefig(path.output_dir + '\\1982-P(c, f, d).png')