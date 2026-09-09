#%%
"""
<One-sentence description of the figure.>

Panels
  A  <what it shows>                     <- <file>.parquet / .csv (columns ...)
  B  <what it shows>                     <- <file>
  C  <what it shows; if a block, say so, e.g. 2 x 4 grid of (x, y) pairs>

Data: results/figure_data/<figure_name>/, written by
utils/runs/prepare_figure_data/<figure_name>.py. Nothing is computed here.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from results.figures.plotting_style import *

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))
from PlotEmbeddings import PlotEmbeddings  # only if the figure has 3D embedding panels

FIGURE = "<figure_name>"
KIND = "sim"            # "sim" -> sim_figs/, "app" -> app_figs/

# ---------------------------------------------------------------------------
# Plotting parameters: the only knobs. Inches unless stated.
# ---------------------------------------------------------------------------
WIDTH = "double"        # "double" (6.5 in) or "single" (3.25 in)
MARGIN_L = 0.55         # y-axis label + tick labels
MARGIN_B = 0.40         # x-axis label + tick labels
MARGIN_T = 0.25         # column titles
MARGIN_R = 0.05
COL_GAP = 0.65          # between panels that both carry a y axis
LEGEND_H = 0.35         # strip for the shared legend
PANEL_H = 2.0

TIMEPOINTS = [150, 1500]          # generations ago, drawn as vertical guides
RENDER_KW = dict(width=1200, height=1200)   # canvas for save_static (probe + final)

# ---------------------------------------------------------------------------
# Data: read, index, nothing else.
# ---------------------------------------------------------------------------
def load_data():
    d = {}
    d["rates"] = pd.read_parquet(figure_data(FIGURE, "rates.parquet"))
    d["pop_labels"] = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
    d["pop_colors"] = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
    params = pd.read_csv(figure_data(FIGURE, "params.csv"))
    d["params"] = dict(zip(params["parameter"], params["value"]))
    d["color_by_pop"] = dict(zip(d["pop_colors"]["population"], d["pop_colors"]["color"]))
    return d


# ---------------------------------------------------------------------------
# Panels: one function each. Each sets its own labels/limits and returns what
# the assembler needs (legend handles, ...).
# ---------------------------------------------------------------------------
def panel_a(ax, d):
    """Coalescent rate through time: truth dashed black, estimate in population colour."""
    handles = []
    for pop, grp in d["rates"].groupby("population"):
        ax.plot(grp["time"], grp["true_rate"], **TRUTH_STYLE)
        ax.plot(grp["time"], grp["estimated_rate"], color=d["color_by_pop"][pop], **ESTIMATE_STYLE)
        handles.append(Line2D([], [], color=d["color_by_pop"][pop], label=pop))
    handles.append(Line2D([], [], label="Truth", **TRUTH_STYLE))
    log_axis(ax, "x")
    log_axis(ax, "y")
    ax.set_xlabel(TIME_LABEL)
    ax.set_ylabel("Coalescent rate")
    return handles


def panel_b(ax, d):
    """3D embeddings through time, rendered by PlotEmbeddings and placed as an image."""
    plotter = PlotEmbeddings(embeddings_df=d["embeddings"], labels_df=d["pop_labels"],
                             colors_df=d["pop_colors"], sample_id_col="individual_id")
    plotter.init_plot(min_time=float(d["params"]["min_time"]),
                      max_time=float(d["params"]["max_time"]))
    plotter.add_sample_trajectories()
    img, _ = plotter.save_static(**RENDER_KW, scale=4)
    place_image(ax, img)


# ---------------------------------------------------------------------------
# Assembly: owns every inch of the figure.
# ---------------------------------------------------------------------------
def assemble(d):
    fig_w = WIDTHS[WIDTH]
    panel_w = (fig_w - MARGIN_L - MARGIN_R - COL_GAP) / 2 - MARGIN_L / 2
    fig_h = MARGIN_T + PANEL_H + MARGIN_B + LEGEND_H
    fig = new_figure(WIDTH, fig_h)

    gs = inch_gridspec(fig,
                       heights=[MARGIN_T, PANEL_H, MARGIN_B, LEGEND_H],
                       widths=[MARGIN_L, panel_w, COL_GAP + MARGIN_L / 2, panel_w, MARGIN_R])
    ax_a = fig.add_subplot(gs.cell(1, 1))
    ax_b = fig.add_subplot(gs.cell(1, 3))
    legend_ax = fig.add_subplot(gs.cell(3, 1))   # spans the legend strip; drawn empty
    legend_ax.set_axis_off()

    handles = panel_a(ax_a, d)
    panel_b(ax_b, d)

    add_panel_label(ax_a, "A", dx=-40, dy=4)      # left of the y-axis label
    add_panel_label(ax_b, "B", dx=0, dy=4)

    fig.legend(handles=handles, loc="center", ncol=len(handles),
               bbox_to_anchor=(0.5, LEGEND_H / 2 / fig_h))
    return fig


#%%
if __name__ == "__main__":
    save_figure(assemble(load_data()), FIGURE, KIND)
# %%
