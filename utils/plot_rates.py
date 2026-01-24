#!/usr/bin/env python3
import pandas as pd
import argparse
import os
from plotting_style import setup_plotting_style
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ______________________________Plotting Params______________________________
INDIVIDUAL_LINE_WIDTH = 0.5
INDIVIDUAL_LINE_ALPHA = 0.2
MEAN_LINE_WIDTH = 1
MEAN_LINE_ALPHA = 0.7
# ___________________________________________________________________________
def plot_rates(rates_df, pop_colors, pop_labels, ylabel="Coalescence Rate"):
    setup_plotting_style()
    populations = pop_labels['population'].unique()

    n_populations = len(populations)
    pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
    times = rates_df.columns.astype(int)

    fig, ax = plt.subplots(1,len(populations), figsize=(3*len(populations),3),sharex=True,sharey=True)
    fig.supylabel(ylabel)
    for i in range(len(populations)):
        if n_populations == 1:  
            ax = [ax]
        ax[i].set_xlabel("Time")
        ax[i].set_title(f"{populations[i]}")
        ax[i].set_xscale("log")
        for j in range(len(populations)):
            pop1 = populations[i]
            pop2 = populations[j]
            if pop1 == pop2:
                color = "black"
            else:
                color = pop_colors_dict[pop2]
            pop1_samples = pop_labels[pop_labels['population'] == pop1]['sample_id'].values
            pop2_samples = pop_labels[pop_labels['population'] == pop2]['sample_id'].values
            sample_pairs = []
            for s1 in pop1_samples:
                for s2 in pop2_samples:
                    if s1 < s2:
                        if (s1,s2) not in sample_pairs:
                            sample_pairs.append((s1,s2))
                    elif s1 > s2:
                        if (s2,s1) not in sample_pairs:
                            sample_pairs.append((s2,s1))
            mean_rates = np.mean(rates_df.loc[sample_pairs,:].values, axis=0)
            ax[i].plot(times, mean_rates, color=color, lw = MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA)
            for pair in sample_pairs:
                ax[i].plot(times, rates_df.loc[pair,:].values, color=color, lw=INDIVIDUAL_LINE_WIDTH, alpha=INDIVIDUAL_LINE_ALPHA)

    legend_handles = [Line2D(
            [0], [0],
            color=color,
            linewidth=3,
            linestyle='-',
            label=pop)
        for pop, color in pop_colors_dict.items()]
    ax[-1].legend(handles=legend_handles, title="Population", bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot coalescence rates")
    parser.add_argument("--input", required=True, help="Path to rates parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    
    args = parser.parse_args()
    rates_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    output_path = args.output_path
    output_name = args.output_name

    # Subsetting labels_df and colors_df to active samples only
    active_pairs = rates_df.index.values
    active_samples = np.unique([item for sublist in active_pairs for item in sublist])
    pop_labels = labels_df[labels_df['sample_id'].isin(active_samples)]
    populations = pop_labels['population'].unique()
    pop_colors = colors_df[colors_df['population'].isin(populations)]

    rates_fig = plot_rates(rates_df=rates_df, pop_colors=pop_colors, pop_labels=pop_labels)
    rates_fig.savefig(os.path.join(output_path, output_name), dpi=500)


if __name__ == "__main__":
    main()

