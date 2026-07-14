#%%
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(os.path.dirname(__file__))
from plotting_style import *
import sys
from tqdm import tqdm
sys.path.append("../../../src")
sys.path.append("../../../utils")
from optimization_functions import optimize_timepoint
from compute_rates_true import compute_true_rates, expand_true_rates
from preprocess_demography import preprocess_demography
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
from PlotEmbeddings import PlotEmbeddings
#%%
deme_file = "../../../data/demes/main/zigzag.yaml"
max_time = 40000
delta_t = 10
n_samples_per_pop = 20
pop_labels, pop_colors = preprocess_demography(deme_file, n_samples_per_pop=n_samples_per_pop)

# Compute true coalescent rates and expand them to match the number of samples
true_rates = compute_true_rates(deme_file, max_time, delta_t)
expanded_true_rates = expand_true_rates(true_rates, pop_labels, phased = False)

# Create a LatentSpaceEmbeddings object and compute the embeddings
N_max = 1e6
N_min_list = [1,100,1000,5000]
for N_min in N_min_list:
    LSE = LatentSpaceEmbeddings(expanded_true_rates, N_min=N_min, N_max=N_max)
    X,loss = LSE.compute_embeddings_through_time()
    embeddings_df = LSE.get_embeddings_df()
    plotter = PlotEmbeddings(embeddings_df = embeddings_df, labels_df = pop_labels, colors_df = pop_colors)
    plotter.init_plot()
    plotter.add_sample_trajectories(alpha = 0.8)
    fig_3d = plotter.fig
    fig_3d.update_layout(showlegend=False)
    plotter.save_static(f"../misc_figs/zigzag_Nmin_{N_min}", format = "png", width = 800, height = 800)
#%%
fig,axs = plt.subplots(1,4,figsize=(6, 1.5))
for i, N_min in enumerate(N_min_list):
    img = plt.imread(f"../misc_figs/zigzag_Nmin_{N_min}.png")
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    axs[i].tick_params(left=False, bottom=False)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].imshow(img, aspect = "equal")
    axs[i].set_title(f"$N_{{\\text{{min}}}} = {N_min}$", fontsize = 8)
filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../outputs/{filename}.png")
# %%
