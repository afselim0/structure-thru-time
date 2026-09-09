#%%
"""
Embeddings of a set of fully separated populations of differing size, under each
link function and p. Block A holds the N_max = 1e5 fits, block B the N_max = 1e6
ones; within each block the top row is the inverse-distance family and the bottom
row the exponential one.

Data: results/figure_data/split_different_links/, written by
utils/runs/prepare_figure_data/split_different_links.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
from results.figures.plotting_style import *

FIGURE = "split_different_links"
embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
populations = pd.read_csv(figure_data(FIGURE, "populations.csv"))

Ne_list = populations["Ne"].tolist()
N_max_values = sorted(embeddings["N_max"].unique())
# %%
colors = plt.cm.viridis(np.linspace(0, 1, len(Ne_list)))
fig = plt.figure(figsize=(6, 7))
gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 0.3, 1, 1])
axs = np.empty((4, 4), dtype=object)
for r in range(4):
    gs_row = r if r < 2 else r + 1
    for c in range(4):
        axs[r, c] = fig.add_subplot(gs[gs_row, c])

# each link/p combination fills one panel per N_max: the first block of two rows
# for the smaller N_max and the second block for the larger. A combination the
# optimizer could not fit is absent from the table and leaves its panel empty.
for (panel_order, N_max), panel in embeddings.groupby(["panel_order", "N_max"], sort=True):
    block = N_max_values.index(N_max)          # 0 -> rows 0-1, 1 -> rows 2-3
    ax = axs[panel_order // 4 + 2 * block, panel_order % 4]
    ax.scatter(panel["X1"], panel["X2"], c=colors[panel["pop_index"].to_numpy()], s=10)
    ax.set_title(f"p={panel['p_label'].iloc[0]}", fontsize=10)

fig.canvas.draw()
row0_left = axs[0, 0].get_position().x0
row0_right = axs[0, 3].get_position().x1
row0_top = axs[0, 0].get_position().y1
row2_top = axs[2, 0].get_position().y1
row_center_x = (row0_left + row0_right) / 2
fig.text(row_center_x, row0_top + 0.05, r"$N_{\text{min}} = 1,\ N_{\text{max}} = 10^5$", ha='center', va='bottom', fontsize=12)
fig.text(row_center_x, row2_top + 0.05, r"$N_{\text{min}} = 1,\ N_{\text{max}} = 10^6$", ha='center', va='bottom', fontsize=12)

fig.text(row0_left - 0.07, (axs[0,0].get_position().y0 + axs[0,0].get_position().y1) / 2, r"$f(x) = (1 + x^p)^{-1}$", ha='right', va='center', rotation=90, fontsize=12)
fig.text(row0_left - 0.07, (axs[2,0].get_position().y0 + axs[2,0].get_position().y1) / 2, r"$f(x) = (1 + x^p)^{-1}$", ha='right', va='center', rotation=90, fontsize=12)
fig.text(row0_left - 0.07, (axs[1,0].get_position().y0 + axs[1,0].get_position().y1) / 2, r"$f(x) = e^{-x^p}$", ha='right', va='center', rotation=90, fontsize=12)
fig.text(row0_left - 0.07, (axs[3,0].get_position().y0 + axs[3,0].get_position().y1) / 2, r"$f(x) = e^{-x^p}$", ha='right', va='center', rotation=90, fontsize=12)

fig.text(row0_left - 0.07, row0_top + 0.05, "A", ha='right', va='bottom', fontsize=12, fontweight='bold')
fig.text(row0_left - 0.07, row2_top + 0.05, "B", ha='right', va='bottom', fontsize=12, fontweight='bold')

# legend mapping Ne values to their viridis color, placed to the right of the grid
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[j], markersize=6, label=f"{Ne}")
    for j, Ne in enumerate(Ne_list)
]
row0_right_all = axs[0, 3].get_position().x1
fig.legend(handles=legend_handles, title=r"$N_e$", loc='center left', bbox_to_anchor=(row0_right_all + 0.02, 0.5), fontsize=9, title_fontsize=10, frameon=False)

file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../sim_figs/{file_name}.pdf", dpi=500)

# %%
