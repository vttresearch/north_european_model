import matplotlib.pyplot as plt
import os
import pandas as pd
from plot_utils import path, fuel, regionCodes, data_processing
import gams.transfer as gt

os.system('cls')

t0 = 0*7*24#(31+28+31+30+31+30)*24
inflowCodes = regionCodes + ['SE01', 'SE03', 'EE00']

# Plot settings ________________________________________________________________________________________________________

def region_subplots(axs, gen_rft, con_rt, fuel_color, inflow_colors, i, region, t0):
    
    ax = axs[i, 0]

    gen = gen_rft.loc[region]
    generation_columns = [column for column in gen.columns if column not in inflowCodes]
    inflow_columns = [column for column in inflowCodes if column in gen.columns]

    generation = gen[generation_columns]
    generation.plot.area(
        stacked=True,
        ax=ax,
        legend=False,
        xlim=[t0, t0 + 168],
        color=fuel_color.iloc[:len(generation_columns)],
        linewidth=0
    )

    # Stack hatched imports above the generation total.
    cumulative = generation.sum(axis=1)
    for source, color in zip(inflow_columns, inflow_colors):
        if source in regionCodes[:4]:
            hatch_pattern = '/' * 3
        elif source in regionCodes[4:]:
            hatch_pattern = '\\' * 3
        else:
            hatch_pattern = 'x' * 3

        next_cumulative = cumulative + gen[source]
        ax.fill_between(
            gen.index,
            cumulative,
            next_cumulative,
            facecolor=color,
            edgecolor='white',
            hatch=hatch_pattern,
            linewidth=0.3,
            label=source
        )
        cumulative = next_cumulative

    con = con_rt.loc[region]
    con.plot(ax=ax, legend=False, xlim=[t0, t0 + 168], color='black')

    ax.text(0.5, 0.85, region, transform=ax.transAxes)


def format_axes(axs, t0):

    day_end = [t0 + x for x in [24, 48, 72, 96, 120, 144, 168]]
    day_mid = [x - 12 for x in day_end]

    for rowNR, row in enumerate(axs):
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
            
            if rowNR < 2:
                ax.set_ylim(0, 24)
            else:
                ax.set_ylim(0, 8)

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

    plt.suptitle('Generation and imports [GW]', fontsize=9)
    plt.subplots_adjust(top=0.96, bottom=0.15, left=0.05, right=0.95, wspace=0, hspace=0)


def plot_generation(gen_rft, con_rt, fuel, regionCodes, t0, path):

    plt.rcParams['font.size'] = 9
    fig, axs = plt.subplots(7, 1, figsize=(17/2.54, 23/2.54), squeeze=False)
    inflow_colors = pd.Series([(0.8, 0.6, 0.0), (0.7, 0.5, 0.0), (0.6, 0.4, 0.0), (0.5, 0.3, 0.0),
                               (0.8, 0.6, 0.0), (0.7, 0.5, 0.0), (0.6, 0.4, 0.0), 
                               (0.6, 0.6, 0.0), (0.5, 0.5, 0.0), (0.4, 0.4, 0.0)])

    for i, region in enumerate(regionCodes):
        region_subplots(axs, gen_rft, con_rt, fuel.color, inflow_colors, i, region, t0)

    format_axes(axs, t0)
    legend_and_labels(fig, axs, fuel.name)

    plt.savefig(os.path.join(path.output_dir, 'P(r, f, h).png'))

# Processing flows _____________________________________________________________________________________________________

def flowdata(path):

    # Loading
    result_gdx = gt.Container(path.debug_gdx)
    tr_gnnft  = result_gdx['r_transfer_gnnft'].records

    # Filtering rows and columns
    tr_gnnft = tr_gnnft[tr_gnnft.grid.str.contains('elec')]
    tr_gnnft = tr_gnnft[tr_gnnft.from_node.str.contains('FI') | tr_gnnft.to_node.str.contains('FI')]

    tr_nnt = tr_gnnft.drop(['grid', 'f'], axis=1)

    # Reindexing
    tr_nnt.from_node = tr_nnt.from_node.apply(lambda s: s.split('_', 1)[0])
    tr_nnt.to_node   = tr_nnt.to_node.  apply(lambda s: s.split('_', 1)[0])
    tr_nnt.t         = tr_nnt.t.str.extract(r'(\d+)').astype(int)

    # Swapping nodes to obtain positive values
    tr_nnt.from_node = tr_nnt.from_node.astype(str)
    tr_nnt.to_node = tr_nnt.to_node.astype(str)

    i_negative = tr_nnt.value < 0
    
    tr_nnt.loc[i_negative, ['from_node', 'to_node']] = tr_nnt.loc[i_negative, ['to_node', 'from_node']].to_numpy()
    tr_nnt.loc[i_negative, 'value'] = -tr_nnt.loc[i_negative, 'value']

    return tr_nnt

def flow_to_fuel(tr_nnt, gen_rft, regionCodes):

    # Filtering and pivoting
    inflow = tr_nnt[tr_nnt.to_node.isin(regionCodes)]
    inflow = inflow.pivot(index=['to_node', 't'], columns='from_node', values='value' ).fillna(0)

    # Reformating to match gen_rft
    inflow.index = inflow.index.set_names(['node', 't'])
    inflow = inflow.reindex(gen_rft.index, fill_value=0)
    inflow /= 1e3

    inflow = inflow[inflowCodes]

    return pd.concat([gen_rft, inflow], axis=1)

# Main _________________________________________________________________________________________________________________

tr_nnt = flowdata(path)

gen_rft, con_rt, fuel = data_processing(path, fuel, regionCodes, 't')
gen_rft = flow_to_fuel(tr_nnt, gen_rft, regionCodes)

fuel.name = pd.concat([fuel.name.iloc[:-1], pd.Series(inflowCodes), fuel.name.iloc[-1:]], ignore_index=True)

plot_generation(gen_rft, con_rt, fuel, regionCodes, t0, path)