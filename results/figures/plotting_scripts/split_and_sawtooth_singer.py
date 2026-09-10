#%%
"""
CoaSTT embeddings from simulated trees against embeddings from SINGER-inferred
trees, for a two-population split (top row) and a single population with a
sawtooth size history (bottom row).

Data: results/figure_data/split_and_sawtooth_singer/, written by
utils/runs/prepare_figure_data/split_and_sawtooth_singer.py -- the SINGER embeddings
there are already Procrustes-aligned to the simulated-tree ones.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
filename = os.path.basename(__file__).split(".")[0]
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
from PlotEmbeddings import *

FIGURE = "split_and_sawtooth_singer"

# rows of the figure, and the title each row carries
DEMOGRAPHIES = ["split", "sawtooth"]
ROW_TITLES = {"split": "Split", "sawtooth": "Sawtooth $N_e$"}
# columns: the tree source, named exactly as the prep script labels it
SOURCES = ["simulated trees", "SINGER trees"]
COLUMN_TITLES = {"simulated trees": "Simulated trees", "SINGER trees": "SINGER trees"}
PANEL_LETTERS = ["A", "B", "C", "D"]

embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
pop_labels = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
pop_colors = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
params = pd.read_csv(figure_data(FIGURE, "params.csv"))


def param(demography, name):
    return params.loc[(params["demography"] == demography)
                      & (params["parameter"] == name), "value"].iloc[0]


# the panel is rendered on this canvas; the cropped aspect ratio depends on the
# canvas size, so the probe and the final render must share it
RENDER_KW = dict(width=1200, height=1200)


def render_panel(demography, source, scale=4):
    """Render one static 3D embeddings panel and return (image, aspect ratio)."""
    df = embeddings[(embeddings["demography"] == demography)
                    & (embeddings["source"] == source)]
    plotter = PlotEmbeddings(
        embeddings_df=df[["individual_id", "time", "X1", "X2"]].copy(),
        labels_df=pop_labels[pop_labels["demography"] == demography],
        colors_df=pop_colors[pop_colors["demography"] == demography],
        sample_id_col="individual_id",
    )
    plotter.init_plot(min_time=float(param(demography, "min_time")),
                      max_time=float(param(demography, "max_time")),
                      time_scale=param(demography, "time_scale"),
                      xy_lims=float(param(demography, "xy_lims")))
    plotter.add_sample_trajectories()
    return plotter.save_static(**RENDER_KW, scale=scale)


#%%
# ---- layout: panel height is fixed, and each panel's width follows its render ----
_, aspect_ratio = render_panel(DEMOGRAPHIES[0], SOURCES[0], scale=1)

FIG_W = 6.5
LABEL_W = 0.30                         # strip on the left holding the rotated row labels
PANEL_H = (FIG_W - LABEL_W) / 2 / aspect_ratio   # two columns filling the rest of the width
TITLE_H = 0.28                         # strip above row 1 holding the column titles
LEGEND_H = 0.45                        # strip below row 2 holding the population key
FIG_H = 2 * PANEL_H + TITLE_H + LEGEND_H

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=500)
outer = gridspec.GridSpec(
    nrows=3, ncols=1, height_ratios=[TITLE_H, 2 * PANEL_H, LEGEND_H], hspace=0,
    left=0, right=1, top=1, bottom=0, figure=fig)
# the row labels get a column of their own, so they never sit on top of a panel
body = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=2, subplot_spec=outer[1],
    width_ratios=[LABEL_W, FIG_W - LABEL_W], wspace=0)
grid = gridspec.GridSpecFromSubplotSpec(
    nrows=2, ncols=2, subplot_spec=body[0, 1], wspace=0, hspace=0)

axes = [[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(2)]

#%%
for i, demography in enumerate(DEMOGRAPHIES):
    for j, source in enumerate(SOURCES):
        ax = axes[i][j]
        img, _ = render_panel(demography, source)
        img.save(f"../app_figs_png/{filename}_{demography}_{source.split()[0]}.png")
        ax.set_axis_off()
        ax.imshow(img, aspect="equal")
        # the rendered panel carries its own white margin, so the letter is placed
        # just outside the axes to keep it clear of the top-left tick label
        ax.text(-0.02, 1.02, PANEL_LETTERS[2 * i + j], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="left")

# column titles sit in the strip above the top row, centred on each column
for j, source in enumerate(SOURCES):
    pos = axes[0][j].get_position()
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.005, COLUMN_TITLES[source],
             ha="center", va="bottom", fontsize=11)

# row titles, rotated in the reserved strip to the left of each row
for i, demography in enumerate(DEMOGRAPHIES):
    pos = axes[i][0].get_position()
    fig.text(pos.x0 - 0.008, (pos.y0 + pos.y1) / 2, ROW_TITLES[demography],
             ha="center", va="center", rotation=90, fontsize=11)

# one population key for the whole figure. Only rows with more than one population
# need distinguishing, so the sawtooth row's single population (named "generic" in
# its demes file, and drawn in the same red as population A) stays out of the key
multi_pop_rows = pop_colors.groupby("demography")["population"].transform("nunique") > 1
populations = pop_colors[multi_pop_rows].drop_duplicates("population")
legend_handles = [Line2D([], [], color=row["color"], lw=3, label=row["population"])
                  for _, row in populations.iterrows()]
fig.legend(handles=legend_handles, title="Population", ncol=len(legend_handles),
           loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=9,
           title_fontsize=10, frameon=False)

fig.savefig(f"../app_figs/{filename}.pdf", bbox_inches="tight", pad_inches=0.02)
# the same figure as a PNG, for quick viewing without a PDF reader
fig.savefig(f"../app_figs_png/{filename}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print(f"Saved ../app_figs/{filename}.pdf")
# %%
