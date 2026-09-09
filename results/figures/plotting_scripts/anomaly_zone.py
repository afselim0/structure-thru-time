#%%
"""
Probability of each unlabelled four-taxon topology under the anomaly-zone
demography: the most frequent topology is not the species tree.

Data: results/figure_data/anomaly_zone/, written by
utils/runs/prepare_figure_data/anomaly_zone.py
"""
import pandas as pd
from results.figures.plotting_style import *
import matplotlib.pyplot as plt
import os
filename = os.path.basename(__file__).split(".")[0]

# Relative paths below are anchored at this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIGURE = "anomaly_zone"
topologies = pd.read_csv(figure_data(FIGURE, "topology_probabilities.csv"))

# the species tree is drawn in red so it stands out among the more probable
# discordant topologies
bar_colors = ["tab:red" if correct else "tab:blue"
              for correct in topologies["is_correct_topology"]]
topology_labels = topologies["topology"].tolist()

fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
ax.bar(topology_labels, topologies["probability"], color=bar_colors)
ax.set_xticks(range(len(topology_labels)))
ax.set_xticklabels(topology_labels, rotation=45, ha="right")
for tick_label, color in zip(ax.get_xticklabels(), bar_colors):
    tick_label.set_color(color)
ax.set_ylabel("Probability")
ax.set_xlabel("Topology")
fig.tight_layout()
fig.savefig(f"../sim_figs/{filename}.pdf", dpi=500)
# %%
