#%%
"""
Average embedded pairwise distance in a panmictic population, against the minimum
resolvable population size N_min (A) and against the number of samples (B).

Data: results/figure_data/avg_distance_vs_Nmin_n/, written by
utils/runs/prepare_figure_data/avg_distance_vs_Nmin_n.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from results.figures.plotting_style import *

# Relative paths below are anchored at this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIGURE = "avg_distance_vs_Nmin_n"
vs_N_min = pd.read_parquet(figure_data(FIGURE, "vs_N_min.parquet"))
vs_n = pd.read_parquet(figure_data(FIGURE, "vs_n.parquet"))

Ne_list = list(dict.fromkeys(vs_N_min["Ne"]))
#%%
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
for Ne in Ne_list:
    left = vs_N_min[vs_N_min["Ne"] == Ne]
    right = vs_n[vs_n["Ne"] == Ne]
    ax[0].plot(left["N_min"], left["avg_pairwise_distance"], label=f"Ne={Ne}")
    ax[1].plot(right["n"], right["avg_pairwise_distance"], label=f"Ne={Ne}")
ax[0].set_xlabel(r"Minimum resolvable $N_e$ ($N_{\mathrm{min}}$)")
ax[0].set_ylabel("Average pairwise distance")
ax[1].set_xlabel("number of samples")
ax[1].set_ylabel("Average pairwise distance")
ax[0].legend()
ax[0].set_title("A", loc='left', fontweight='bold')
ax[1].set_title("B", loc='left', fontweight='bold')
fig.suptitle(r"$f(x) = (N_{\mathrm{min}} + x)^{-1}$")
file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../sim_figs/{file_name}.pdf", dpi=500)
# %%
