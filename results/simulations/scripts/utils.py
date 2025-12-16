#%%
import os
import demes
import argparse
import msprime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
#%%
def setup_plotting_style():
    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5
    # -- Fonts --
    rcParams['font.size'] = 10
    rcParams['font.family'] = 'serif'
    # plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    # -- Figure size --
    rcParams['figure.figsize'] = (6, 6)
    # -- Saving Options --
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.dpi'] = 500
#%%
def preprocess_demography(demes_file,n_samples_per_pop):
    graph = demes.load(demes_file)
    demography = msprime.Demography.from_demes(graph)
    pops = [pop.name for pop in demography.populations if pop.default_sampling_time == 0]
    if len(pops) <= 9:
        cmap = plt.get_cmap("Set1")
        pop_colors_dict = {pop:mcolors.to_hex(cmap(i)[:3]) for i,pop in enumerate(pops)}
    else:
        cmap = plt.get_cmap("rainbow")
        color_idx = np.linspace(0,1,len(pops))
        pop_colors_dict = {pop:mcolors.to_hex(cmap(color_idx[i])[:3]) for i,pop in enumerate(pops)}
    pop_colors = pd.DataFrame({
        "population": list(pop_colors_dict.keys()),
        "color": list(pop_colors_dict.values())
    })
    pop_labels = np.repeat(pops, n_samples_per_pop)
    pop_labels = pd.DataFrame({
        "sample_id": np.arange(len(pop_labels)),
        "population": pop_labels
    })
    return pop_labels, pop_colors
# %%
