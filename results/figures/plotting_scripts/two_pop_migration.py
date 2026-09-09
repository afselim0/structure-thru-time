#%%
"""
Two populations splitting, under recent migration (row A) and a soft split
(row B), at four scaled migration rates.

Data: results/figure_data/two_pop_migration/, written by
utils/runs/prepare_figure_data/two_pop_migration.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
from results.figures.plotting_style import *
import sys
sys.path.append("../../../src")
from PlotEmbeddings import PlotEmbeddings

FIGURE = "two_pop_migration"
output_path = "../sim_figs_png/"
os.makedirs(output_path, exist_ok=True)

embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
pop_labels = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
pop_colors = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))

# one column per scaled migration rate, weakest first
Nm_list = sorted(embeddings["Nm"].unique())
SCENARIOS = ["early", "deep"]   # row 0: recent migration, row 1: soft split
#%%
fig, axs = plt.subplots(2, 4, figsize=(6, 3))
for row, scenario in enumerate(SCENARIOS):
    for col, Nm in enumerate(Nm_list):
        panel = embeddings[(embeddings["scenario"] == scenario)
                           & (embeddings["Nm"] == Nm)]
        plotter = PlotEmbeddings(embeddings_df=panel, labels_df=pop_labels,
                                 colors_df=pop_colors)
        plotter.init_plot(time_scale="linear")
        plotter.add_sample_trajectories(alpha=0.8)
        plotter.fig.update_layout(showlegend=False)
        img, _ = plotter.save_static(width=800, height=800)
        img.save(f"{output_path}{scenario}_mig_{Nm}.png")

        ax = axs[row, col]
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(left=False, bottom=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, aspect="equal")
        if row == 0:
            ax.set_title(f"$Nm = {Nm}$", fontsize=8)

fig.text(axs[0,0].yaxis.label.get_position()[0]-0.01, 1.01, "A) Recent migration", fontweight="bold", fontsize = 11)
fig.text(axs[0,1].yaxis.label.get_position()[0]-0.01, 0.5, "B) Soft split", fontweight="bold", fontsize = 11)

filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../sim_figs/{filename}.pdf")
# %%
