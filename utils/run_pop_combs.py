#%%
import argparse
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.io as pio
from itertools import combinations
import sys
from tqdm import tqdm
sys.path.append("../src")
sys.path.append("../../src")
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
from PlotEmbeddings import PlotEmbeddings

def find_active_pops(rates_df, pop_labels, pop_colors):
    active_pairs = rates_df.index.values
    active_samples = np.unique([item for sublist in active_pairs for item in sublist])
    pop_labels = pop_labels[pop_labels['sample_id'].isin(active_samples)]
    populations = pop_labels['population'].unique()
    pop_colors = pop_colors[pop_colors['population'].isin(populations)]
    return populations, pop_labels, pop_colors

def run_pop_combinations(data_dir, plots_dir, pop_labels, pop_colors, rates, N0, generation_time=None):
    comb_data_dir =os.path.join(data_dir,"combinations")
    comb_plot_dir = os.path.join(plots_dir,"combinations")

    os.makedirs(comb_data_dir, exist_ok=True)
    os.makedirs(comb_plot_dir, exist_ok=True)

    populations, pop_labels, pop_colors = find_active_pops(rates, pop_labels, pop_colors)
    population_combinations = []
    for r in range(1, len(populations) + 1):
        population_combinations.extend([list(comb) for comb in combinations(populations, r)])

    for population_combination in tqdm(population_combinations):
        filename = f"{len(population_combination)}_{'_'.join(population_combination)}"

        label_subset = pop_labels[pop_labels['population'].isin(population_combination)]
        colors_subset = pop_colors[pop_colors['population'].isin(population_combination)]
        sample_ids = label_subset['sample_id'].values
        rates_subset = rates.loc[rates.index.get_level_values(0).isin(sample_ids) & rates.index.get_level_values(1).isin(sample_ids)]
        LSE = LatentSpaceEmbeddings(rates_subset, N0=N0)
        _ = LSE.compute_embeddings_through_time()
        embeddings_df = LSE.get_embeddings_df()
        embeddings_df.to_parquet(f"{comb_data_dir}/{filename}.parquet")

        plotter = PlotEmbeddings(embeddings_df, label_subset, colors_subset, generation_time=generation_time)
        plotter.init_plot(time_scale="log")
        plotter.add_sample_trajectories()
        plotter.add_population_medians()
        plotter.add_legend()
        plotter.save_static(filename=f"{comb_plot_dir}/{filename}", format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1))
        plotter.fig.write_html(f"{comb_plot_dir}/{filename}.html")

def main():
    argparser = argparse.ArgumentParser(description="Run latent space embeddings and plot for all population combinations.")
    argparser.add_argument('--data_dir', type=str, required=True, help='Directory containing input data.')
    argparser.add_argument('--plots_dir', type=str, required=True, help='Directory to save output plots.')
    argparser.add_argument('--pop_labels', type=str, required=True, help='Path to population labels CSV file.')
    argparser.add_argument('--pop_colors', type=str, required=True, help='Path to population colors CSV file.')
    argparser.add_argument('--rates_file', type=str, required=True, help='Path to rates parquet file.')
    argparser.add_argument('--N0', type=int, required=True, help='Effective population size.')
    argparser.add_argument('--generation_time', type=int, help='Generation time in years.')
    args = argparser.parse_args()
    data_dir = args.data_dir
    plots_dir = args.plots_dir
    N0 = args.N0
    generation_time = args.generation_time
    pop_labels = pd.read_csv(args.pop_labels)
    pop_colors = pd.read_csv(args.pop_colors)
    rates = pd.read_parquet(args.rates_file)

    run_pop_combinations(data_dir, plots_dir, pop_labels, pop_colors, rates, N0, generation_time)