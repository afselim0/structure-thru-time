#%%
"""
Embeddings of a panmictic population under each link function and p: the top row
is the inverse-distance family, the bottom row the exponential one.

Data: results/figure_data/panmictic_diff_links/, written by
utils/runs/prepare_figure_data/panmictic_diff_links.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from results.figures.plotting_style import *

# Relative paths below are anchored at this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIGURE = "panmictic_diff_links"
embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
#%%
fig, axs = plt.subplots(2, 4, figsize=(6, 3.5))
# panel_order is the position each link/p combination occupies in the 2x4 grid;
# a combination the optimizer could not fit is absent and leaves its panel empty
for i, (panel_order, panel) in enumerate(embeddings.groupby("panel_order", sort=True)):
    ax = axs[panel_order // 4, panel_order % 4]
    ax.scatter(panel["X1"], panel["X2"], s=10)
    ax.set_title(f"p={panel['p_label'].iloc[0]}", fontsize=10)
fig.text(-0.01, 0.78, r"$f(x) = (1 + x^p)^{-1}$", ha='center', va='center', fontsize=12, rotation=90)
fig.text(-0.01, 0.28, r"$f(x) = e^{-x^p}$", ha='center', va='center', fontsize=12, rotation=90)

file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../sim_figs/{file_name}.pdf", dpi=500)
# %%
