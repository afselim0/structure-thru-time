#%%
#!/usr/bin/env python3
import pandas as pd
import argparse
import os
from plotting_style import setup_plotting_style
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ______________________________Plotting Params______________________________
INDIVIDUAL_LINE_WIDTH = 0.2
INDIVIDUAL_LINE_ALPHA = 0.2
MEAN_LINE_WIDTH = 2
MEAN_LINE_ALPHA = 1
# ___________________________________________________________________________
def plot_rates_multi_figs(rates_df, pop_colors, pop_labels, min_time=0, max_rate_plot = None, ylabel="Coalescence Rate", time_scale="linear", phased=False):
    if phased:
        id_label = "sample_id"
    else:
        id_label = "individual_id"
    setup_plotting_style()
    populations = pop_labels['population'].unique()

    n_populations = len(populations)
    pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
    times = rates_df.columns.astype(float)

    fig, ax = plt.subplots(1,len(populations), figsize=(6*len(populations),4),sharex=True,sharey=True)
    fig.supylabel(ylabel)
    for i in range(len(populations)):
        if n_populations == 1:  
            ax = [ax]
        ax[i].set_xlabel("Time (generations ago)")
        ax[i].set_title(f"{populations[i]}")
        ax[i].set_xscale(time_scale)
        if max_rate_plot is not None:
            ax[i].set_ylim(top=max_rate_plot)
        ax[i].set_xlim(float(min_time), times.max())
        for j in range(len(populations)):
            pop1 = populations[i]
            pop2 = populations[j]
            if pop1 == pop2:
                color = "grey"
            else:
                color = pop_colors_dict[pop2]
            pop1_samples = pop_labels[pop_labels['population'] == pop1][id_label].values
            pop2_samples = pop_labels[pop_labels['population'] == pop2][id_label].values
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
            for pair in sample_pairs:
                ax[i].plot(times, rates_df.loc[pair,:].values, color=color, lw=INDIVIDUAL_LINE_WIDTH, alpha=INDIVIDUAL_LINE_ALPHA)
            ax[i].plot(times, mean_rates, color="black" if pop1 == pop2 else color, lw = MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA)

    legend_handles = [Line2D(
            [0], [0],
            color=color,
            linewidth=3,
            linestyle='-',
            label=pop)
        for pop, color in pop_colors_dict.items()]
    ax[-1].legend(handles=legend_handles, title="Population", bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout(rect=[0.01, 0, 1, 1])

    return fig

def plot_rates_single_fig(rates_df, min_time = 0, max_rate_plot = None, ylabel="Coalescence Rate", time_scale="linear"):
    setup_plotting_style()
    times = rates_df.columns.astype(float)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_xlabel("Time (generations ago)")
    ax.set_ylabel(ylabel)
    ax.set_xscale(time_scale)
    if max_rate_plot is not None:
        ax.set_ylim(top=max_rate_plot)
    ax.set_xlim(float(min_time), times.max())
    for pair in rates_df.index.values:
        ax.plot(times, rates_df.loc[pair,:].values, color="grey", lw=INDIVIDUAL_LINE_WIDTH, alpha=INDIVIDUAL_LINE_ALPHA)
    mean_rates = np.mean(rates_df.values, axis=0)
    ax.plot(times, mean_rates, color="black", lw=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot coalescence rates")
    parser.add_argument("--input", required=True, help="Path to rates parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--min_time", type=float, required=True, help="Minimum time for plotting")
    parser.add_argument("--max_rate_plot", type=float, required=True, help="Maximum rate for plotting")
    parser.add_argument("--time_scale", default="linear", help="Time scale for demography plot, linear or log", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    parser.add_argument("--phased", type=str, default="False", help="Indicates whether the input data is phased. If false, colors are based on individual_id rather than sample_id")
    
    args = parser.parse_args()
    rates_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    output_path = args.output_path
    output_name = args.output_name
    min_time = args.min_time
    max_rate_plot = args.max_rate_plot
    phased = args.phased == "True"

    print(f"Plotting coalescence rates from file: {args.input}")

    # Subsetting labels_df and colors_df to active samples only
    if phased:
        id_label = "sample_id"
    else:
        id_label = "individual_id"
    active_pairs = rates_df.index.values
    active_samples = np.unique([item for sublist in active_pairs for item in sublist])
    pop_labels = labels_df[labels_df[id_label].isin(active_samples)]
    populations = pop_labels['population'].unique()
    pop_colors = colors_df[colors_df['population'].isin(populations)]

    if len(populations) <= 6 and len(populations) > 1:
        rates_fig = plot_rates_multi_figs(rates_df=rates_df, pop_colors=pop_colors, pop_labels=pop_labels, min_time=min_time, max_rate_plot=max_rate_plot, time_scale=args.time_scale, phased=phased)
        rates_fig_single = plot_rates_single_fig(rates_df=rates_df, min_time=min_time, max_rate_plot=max_rate_plot, time_scale=args.time_scale)
        rates_fig_single.savefig(os.path.join(output_path, output_name.replace(".png", "_single.png")), dpi=500)
    else:
        rates_fig = plot_rates_single_fig(rates_df=rates_df, min_time=min_time, max_rate_plot=max_rate_plot, time_scale=args.time_scale)
    rates_fig.savefig(os.path.join(output_path, output_name), dpi=500)


if __name__ == "__main__":
    main()
