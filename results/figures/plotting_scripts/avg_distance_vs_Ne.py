#%%
"""
Average embedded pairwise distance in a panmictic population against Ne, for the
two link families and a range of p.

Data: results/figure_data/avg_distance_vs_Ne/, written by
utils/runs/prepare_figure_data/avg_distance_vs_Ne.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from results.figures.plotting_style import *

# Relative paths below are anchored at this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIGURE = "avg_distance_vs_Ne"
distances = pd.read_parquet(figure_data(FIGURE, "distances.parquet"))
#%%
fig, axs = plt.subplots(1, 2, figsize=(6, 3.5), sharex=True, sharey=False)
# panel layout: cols = link family, showing average pairwise distance only
panels = {
    "inverse_distance": axs[0],
    "exponential": axs[1],
}

for (link_name, p), group in distances.groupby(["link", "p"], sort=False):
    ax = panels[link_name]
    ax.plot(group["Ne"], group["avg_pairwise_distance"],
            label=f"p={group['p_label'].iloc[0]}")

for link_name, ax in panels.items():
    ax.set_xlabel("Effective population size (Ne)")
    ax.set_ylabel("Average pairwise distance")
    ax.set_xscale('log')
    ax.set_yscale('log')
axs[0].plot([1e2, 1e4], [1e2, 1e4], 'k--', label="y=x")
axs[0].set_title(r"$f(x) = (1 + x^p)^{-1}$")
axs[1].set_title(r"$f(x) = e^{-x^p}$")
axs[0].legend()
axs[1].legend()
fig.tight_layout()
file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../sim_figs/{file_name}.pdf", dpi=500)
# %%
