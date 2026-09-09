#%%
"""
Two populations splitting 2000 generations ago, embedded through time under two
migration scenarios at four scaled migration rates Nm.

Panels
  A  row of four 3D embedding panels, recent migration (from the present back to
     500 generations ago), one column per Nm            <- embeddings.parquet, scenario == "early"
  B  same, soft split (migration from the split until
     1500 generations ago)                              <- embeddings.parquet, scenario == "deep"
  Shared population legend below.

Data: results/figure_data/two_pop_migration/, written by
utils/runs/prepare_figure_data/two_pop_migration.py. Nothing is computed here.
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from results.figures.plotting_style import *

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))
from PlotEmbeddings import PlotEmbeddings

FIGURE = "two_pop_migration"
KIND = "sim"

# ---------------------------------------------------------------------------
# Plotting parameters: the only knobs. Inches unless stated.
# ---------------------------------------------------------------------------
WIDTH = "double"
ROW_LABEL_W = 0.52     # strip on the left: rotated two-line row titles + the shared time arrow
TITLE_H = 0.20         # strip above row A holding the Nm column titles
ROW_GAP = 0.06         # between the two rows of panels
LEGEND_H = 0.28        # strip below row B holding the population key

SCENARIOS = ["early", "deep"]                 # row order
ROW_TITLES = {"early": "Recent\nmigration", "deep": "Soft\nsplit"}   # two lines: a row is only ~1.2 in tall
ROW_LETTERS = {"early": "A", "deep": "B"}
TIME_SCALE = "linear"

# the 3D renders are small multiples (~1.5 in): Plotly's tick labels cannot be
# made legible at that size, so they are hidden and the time direction is given
# once by the arrow in the left strip. The axis planes stay so every panel crops
# to the same aspect ratio.
RENDER_KW = dict(width=700, height=700)
RENDER_SCALE = 3
TIME_ARROW_LABEL = TIME_LABEL


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    d = {}
    d["embeddings"] = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
    d["pop_labels"] = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
    d["pop_colors"] = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
    params = pd.read_csv(figure_data(FIGURE, "params.csv"))
    d["params"] = dict(zip(params["parameter"], params["value"]))
    d["Nm_list"] = sorted(d["embeddings"]["Nm"].unique())          # weakest first
    # one shared xy range so the eight panels are comparable and share an aspect ratio
    d["xy_lims"] = float(np.abs(d["embeddings"][["X1", "X2"]].to_numpy()).max()) * 1.05
    return d


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def render_embeddings(d, scenario, Nm, scale=RENDER_SCALE):
    """Static 3D trajectories for one (scenario, Nm); returns (image, aspect ratio)."""
    panel = d["embeddings"][(d["embeddings"]["scenario"] == scenario)
                            & (d["embeddings"]["Nm"] == Nm)]
    plotter = PlotEmbeddings(embeddings_df=panel, labels_df=d["pop_labels"],
                             colors_df=d["pop_colors"])
    plotter.init_plot(time_scale=TIME_SCALE, min_time=0.0,
                      max_time=float(d["params"]["max_time"]), xy_lims=d["xy_lims"])
    plotter.add_sample_trajectories(alpha=0.8)
    hidden = dict(showticklabels=False, title="")
    plotter.fig.update_layout(scene=dict(xaxis=hidden, yaxis=hidden, zaxis=hidden))
    return plotter.save_static(**RENDER_KW, scale=scale)


def panel_embeddings(ax, d, scenario, Nm):
    img, _ = render_embeddings(d, scenario, Nm)
    place_image(ax, img)


def population_legend_handles(d):
    return [Line2D([], [], color=c, linewidth=1.5, label=f"Population {p}")
            for p, c in zip(d["pop_colors"]["population"], d["pop_colors"]["color"])]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble(d):
    n_cols = len(d["Nm_list"])
    fig_w = WIDTHS[WIDTH]
    # every render shares canvas and limits, so one probe gives the aspect ratio of all
    _, aspect = render_embeddings(d, SCENARIOS[0], d["Nm_list"][0], scale=1)
    panel_w = (fig_w - ROW_LABEL_W) / n_cols
    panel_h = panel_w / aspect
    fig_h = TITLE_H + 2 * panel_h + ROW_GAP + LEGEND_H
    fig = new_figure(WIDTH, fig_h)

    gs = inch_gridspec(fig,
                       heights=[TITLE_H, panel_h, ROW_GAP, panel_h, LEGEND_H],
                       widths=[ROW_LABEL_W] + [panel_w] * n_cols)
    row_index = {"early": 1, "deep": 3}

    # left strip: one axis spanning both rows holds the rotated row titles (far
    # left) and a single time arrow (next to the panels), since the 3D panels
    # carry no tick labels of their own
    strip_ax = fig.add_subplot(gs[1:4, 0])
    strip_ax.set_axis_off()
    strip_ax.set_xlim(0, 1)
    strip_ax.set_ylim(0, 2 * panel_h + ROW_GAP)
    strip_ax.annotate("", xy=(0.88, 2 * panel_h + ROW_GAP), xytext=(0.88, 0),
                      arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black",
                                      shrinkA=0, shrinkB=0, mutation_scale=7))
    strip_ax.text(0.74, (2 * panel_h + ROW_GAP) / 2, TIME_ARROW_LABEL, rotation=90,
                  ha="center", va="center", fontsize=FONT_SIZES["annotation"])
    for scenario, y0 in zip(SCENARIOS, [panel_h + ROW_GAP, 0.0]):
        strip_ax.text(0.30, y0 + panel_h / 2, ROW_TITLES[scenario], rotation=90, linespacing=1.0,
                      ha="center", va="center", fontsize=FONT_SIZES["title"])

    for scenario in SCENARIOS:
        r = row_index[scenario]
        for j, Nm in enumerate(d["Nm_list"]):
            ax = fig.add_subplot(gs.cell(r, j + 1))
            panel_embeddings(ax, d, scenario, Nm)
            if scenario == SCENARIOS[0]:
                ax.set_title(f"$Nm = {Nm:g}$", fontsize=FONT_SIZES["title"], pad=2)
            if j == 0:
                add_panel_label(ax, ROW_LETTERS[scenario], dx=2, dy=0)

    legend_ax = fig.add_subplot(gs.cell(4, 1))
    legend_ax.set_axis_off()
    fig.legend(handles=population_legend_handles(d), loc="center", ncol=2,
               bbox_to_anchor=((ROW_LABEL_W + (fig_w - ROW_LABEL_W) / 2) / fig_w,
                               LEGEND_H / 2 / fig_h),
               handlelength=1.8, columnspacing=1.5)
    return fig


#%%
if __name__ == "__main__":
    save_figure(assemble(load_data()), FIGURE, KIND)
# %%
