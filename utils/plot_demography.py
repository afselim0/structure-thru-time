#%%
#!/usr/bin/env python3
import os
import argparse
from plotting_style import setup_plotting_style
import pandas as pd
import demes
import demesdraw
import matplotlib.pyplot as plt

def plot_demography(demes_file_path, pop_colors, max_time, min_time):
    setup_plotting_style()
    graph = demes.load(demes_file_path)
    pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
    fig, ax = plt.subplots(figsize=(2, 4.5))
    demesdraw.tubes(
        graph,
        ax=ax,
        colours=pop_colors_dict,
        max_time=max_time,
        log_time=True
    )
    ax.set_ylim(bottom=min_time)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Preprocess demography from a demes file")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file", type=str)
    parser.add_argument("--colors", required=True, help="Path to population colors csv file", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for preprocessed demography data", type=str)
    parser.add_argument("--max_time", required=True, help="Maximum time for demography plot", type=int)
    parser.add_argument("--min_time", required=True, help="Minimum time for demography plot", type=int)

    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    colors = args.colors
    output_path = args.output_path
    max_time = args.max_time
    min_time = args.min_time
    pop_colors = pd.read_csv(colors)
    demes_fig = plot_demography(demes_file_path, pop_colors, max_time, min_time)

    demes_fig.savefig(os.path.join(output_path, "demography.png"), dpi=500)

if __name__ == "__main__":
    main()