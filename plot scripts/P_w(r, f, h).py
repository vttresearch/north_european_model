import matplotlib.pyplot as plt
import os
from plot_utils import path, fuel, regionCodes, data_processing

os.system('cls')

t0 = 0*7*24#(31+28+31+30+31+30)*24

# Plot settings ________________________________________________________________________________________________________

def region_subplots(axs, gen_rft, con_rt, fuel_color, i, region, t0):
    
    ax = axs[i, 0]

    gen = gen_rft.loc[region]
    gen.plot.area(stacked=True, ax=ax, legend=False, xlim=[t0, t0 + 168], color=fuel_color, linewidth=0)

    con = con_rt.loc[region]
    con.plot(ax=ax, legend=False, xlim=[t0, t0 + 168], color='black')

    ax.text(0.5, 0.85, region, transform=ax.transAxes)


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


def plot_generation(gen_rft, con_rt, fuel, regionCodes, t0, path):

    plt.rcParams['font.size'] = 9
    fig, axs = plt.subplots(7, 1, figsize=(17/2.54, 23/2.54), squeeze=False)

    for i, region in enumerate(regionCodes):
        region_subplots(axs, gen_rft, con_rt, fuel.color, i, region, t0)

    format_axes(axs, t0)
    legend_and_labels(fig, axs, fuel.name)

    plt.savefig(os.path.join(path.output_dir, 'P(r, f, h).png'))

# Main _________________________________________________________________________________________________________________

gen_rft, con_rt, fuel = data_processing(path, fuel, regionCodes, 't')

plot_generation(gen_rft, con_rt, fuel, regionCodes, t0, path)