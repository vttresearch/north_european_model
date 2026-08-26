import matplotlib.pyplot as plt
import os
from plot_utils import path, fuel, countryCodes, data_processing

os.system('cls')

t0 = 0*7*24#(31+28+31+30+31+30)*24

# Plot settings ________________________________________________________________________________________________________

def country_subplots(axs, gen_cft, con_ct, fuel_color, i, country, t0):
    
    ax = axs[i % 7, i // 7]

    gen = gen_cft.loc[country]
    gen.plot.area(stacked=True, ax=ax, legend=False, xlim=[t0, t0 + 168], color=fuel_color, linewidth=0)

    con = con_ct.loc[country]
    con.plot(ax=ax, legend=False, xlim=[t0, t0 + 168], color='black')

    ax.text(0.5, 0.85, country, transform=ax.transAxes)


def format_axes(axs, t0):

    day_end = [t0 + x for x in [24, 48, 72, 96, 120, 144, 168]]
    day_mid = [x - 12 for x in day_end]

    for row in axs:
        for columnNr, ax in enumerate(row):
            
            # OX: ticks by hours
            
            ax.set_xticks(day_end, [])
            
            xlabel_y = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
            for mid, label in zip(day_mid, ['1', '2', '3', '4', '5', '6', '7']):
                ax.text(mid, xlabel_y, label, ha='center', va='top')

            ax.set_xlabel('days', labelpad=8)

            # OY: uniform tick number (not always work) without excesive labels

            if columnNr == 1:
                ax.yaxis.tick_right()

            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_ylim(0, ax.get_yticks()[-1])

            tick_labels = ax.get_yticklabels()
            for idx in [0, 1, 3]:
                if idx < len(tick_labels):
                    tick_labels[idx].set_visible(False)

            # Gridlines

            ax.grid(True, which='major', axis='both', linestyle=':', color='black', linewidth=0.5)
            ax.grid(True, which='minor', axis='y'   , linestyle=':', color='black', linewidth=0.5)


def legend_and_labels(fig, axs, fuel_name):
    handles, labels = axs[0, 0].get_legend_handles_labels()
    leg = fig.legend(handles[::-1], fuel_name[::-1], loc='lower center', ncol=6, handlelength=1, handleheight=1, columnspacing=0.6)
    leg.get_frame().set_edgecolor('none')

    plt.suptitle('Generation [GW]', fontsize=9)
    plt.subplots_adjust(top=0.96, bottom=0.12, left=0.05, right=0.95, wspace=0, hspace=0)


def plot_generation(gen_cft, con_ct, fuel, countryCodes, t0, path):

    plt.rcParams['font.size'] = 9
    fig, axs = plt.subplots(7, 2, figsize=(17/2.54, 23/2.54))

    for i, country in enumerate(countryCodes):
        country_subplots(axs, gen_cft, con_ct, fuel.color, i, country, t0)

    format_axes(axs, t0)
    legend_and_labels(fig, axs, fuel.name)

    plt.savefig(os.path.join(path.output_dir, 'P(c, f, h).png'))

# Main _________________________________________________________________________________________________________________

gen_cft, con_ct, fuel = data_processing(path, fuel, countryCodes, 't')

plot_generation(gen_cft, con_ct, fuel, countryCodes, t0, path)