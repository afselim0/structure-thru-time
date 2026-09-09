#%%
"""
Coalescence rates under symmetric migration between two populations.

Writes two figures: the within/between rate trajectories at four Nm values
(mig_coal_rate.pdf), and how the between-population plateau and its rate constant
scale with Nm (mig_coal_rate_2.pdf).

Data: results/figure_data/coal_rate_mig_rate/, written by
utils/runs/prepare_figure_data/coal_rate_mig_rate.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from results.figures.plotting_style import *
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIGURE = "coal_rate_mig_rate"
trajectories = pd.read_parquet(figure_data(FIGURE, "rate_trajectories.parquet"))
saturation = pd.read_parquet(figure_data(FIGURE, "saturation.parquet"))

Nm_list = list(dict.fromkeys(trajectories["Nm"]))
Nm_labels = dict(zip(trajectories["Nm"], trajectories["Nm_label"]))
#%%
# Between and within coalescence rates over time for different Nm values
fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.75), sharey=True, dpi=500)
for i, Nm in enumerate(Nm_list):
    panel = trajectories[trajectories["Nm"] == Nm]
    axs[i].plot(panel["time"], panel["between_rate"], label='Between')
    axs[i].plot(panel["time"], panel["within_rate"], label='Within')
    axs[i].set_title(f'Nm = {Nm_labels[Nm]}', fontsize=10)
fig.supxlabel('Time (generations ago)', fontsize=10)
axs[0].set_ylabel('Coalescence rate', fontsize=10)
axs[-1].legend(fontsize=8)
fig.savefig("../sim_figs/mig_coal_rate.pdf", bbox_inches='tight', dpi=500)
#%%
# Saturation limit and rate of convergence for different Nm values
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.75), dpi=500)
axs[0].plot(saturation["Nm"], saturation["plateau"])
axs[1].plot(saturation["Nm"], saturation["rate_constant"])

axs[0].set_xlabel('Nm', fontsize=8)
axs[0].set_ylabel('Limiting coalescent rate', fontsize=8)
axs[1].set_xlabel('Nm', fontsize=8)
axs[1].set_ylabel('Rate constant', fontsize=8)
axs[0].set_title('A', loc='left', fontweight='bold', fontsize=8)
axs[1].set_title('B', loc='left', fontweight='bold', fontsize=8)
axs[0].tick_params(axis='both', which='major', labelsize=8)
axs[1].tick_params(axis='both', which='major', labelsize=8)
fig.savefig("../sim_figs/mig_coal_rate_2.pdf", bbox_inches='tight', dpi=500)
# %%
