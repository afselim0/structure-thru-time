#%%
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils/snakemake_utils")
sys.path.append("../../../utils/misc_utils")
from plot_rates import compute_mean_rates
#%%
rates_df = pd.read_parquet("../data/rates.minWS70.maxWS4000.V1.FS1.0.maxT20000.nT5000.parquet")
pop_labels = pd.read_csv("../../../data/metadata/wohns_pop_labels.csv")
active_pairs = rates_df.index.values
active_samples = np.unique([item for sublist in active_pairs for item in sublist])
pop_labels = pop_labels[pop_labels["individual_id"].isin(active_samples)]
mean_rates, populations = compute_mean_rates(rates_df, pop_labels, phased=False)
times = rates_df.columns.astype(float)
times *= 28
within_pop_colors_dict = {
    "YRI": '#0072B2',
    "CEU": '#D55E00',
    "CHB": '#009E73'}
between_pop_colors_dict = {
    "YRI-CEU": '#CC79A7',
    "YRI-CHB": '#56B4E9',
    "CEU-CHB": '#E69F00'}
# %%
fig,axs = plt.subplots(1, 3, figsize=(6,2), sharey=True)
within_pop_ax = axs[0]
between_pop_ax = axs[1]
all_rates_ax = axs[2]

# A) mean within population rates
for pop,color in within_pop_colors_dict.items():
    pop_idx = np.where(populations == pop)[0][0]
    within_pop_ax.plot(times, mean_rates[pop_idx,pop_idx,:], color=color, label=pop)

within_pop_ax.legend(loc="lower right", fontsize=8, frameon=False)
# B) mean between lower rates
for pop_comb,color in between_pop_colors_dict.items():
    pop1, pop2 = pop_comb.split("-")
    pop1_idx = np.where(populations == pop1)[0][0]
    pop2_idx = np.where(populations == pop2)[0][0]
    between_pop_ax.plot(times, mean_rates[pop1_idx,pop2_idx,:], color=color, label=pop_comb, lw=2)

between_pop_ax.legend(loc="lower right", fontsize=8, frameon=False)
# C) All rates
for _,row in rates_df.iterrows():
    all_rates_ax.plot(times, row, c= "#AEABAC", lw = 0.1)


# use log scale with scientific notation for x-axis tick labels and add minor ticks on all subplots
for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1000,)
    ax.set_ylim(0.2e-8, 5e-4)
    ax.set_xlabel("Time (years ago)")
axs[0].set_ylabel("Coalescent Rate")

axs[0].set_title("A", loc="left", fontweight="bold")
axs[1].set_title("B", loc="left", fontweight="bold")
axs[2].set_title("C", loc="left", fontweight="bold")

filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../app_figs/{filename}.pdf")
# %%
