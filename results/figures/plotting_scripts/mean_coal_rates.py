#%%
"""
Mean coalescence rates in the Wohns human dataset: within populations (A), between
populations (B), and every individual pair (C).

Data: results/figure_data/mean_coal_rates/, written by
utils/runs/prepare_figure_data/mean_coal_rates.py -- times there are already in
years.
"""
import matplotlib.pyplot as plt
import pandas as pd
from results.figures.plotting_style import *
import os
os.chdir(os.path.dirname(__file__))

FIGURE = "mean_coal_rates"
within_rates = pd.read_parquet(figure_data(FIGURE, "within_population_rates.parquet"))
between_rates = pd.read_parquet(figure_data(FIGURE, "between_population_rates.parquet"))
all_rates = pd.read_parquet(figure_data(FIGURE, "all_pair_rates.parquet"))

times = within_rates["time"]
all_times = all_rates.columns.astype(float)

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
    within_pop_ax.plot(times, within_rates[pop], color=color, label=pop)

within_pop_ax.legend(loc="lower right", fontsize=8, frameon=False)
# B) mean between population rates
for pop_comb,color in between_pop_colors_dict.items():
    between_pop_ax.plot(times, between_rates[pop_comb], color=color, label=pop_comb, lw=2)

between_pop_ax.legend(loc="lower right", fontsize=8, frameon=False)
# C) All rates
for _,row in all_rates.iterrows():
    all_rates_ax.plot(all_times, row, c= "#AEABAC", lw = 0.1)


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
