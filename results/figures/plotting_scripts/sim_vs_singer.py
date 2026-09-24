#%%
"""
CoaSTT embeddings on simulated trees (left) versus SINGER-inferred trees (right),
for four demographies.

Panels (one row each, 3D embeddings, SINGER Procrustes-aligned onto simulated)
  A  two-population split                     <- embeddings.parquet, demography == "split"
  B  one population, sawtooth size history    <- demography == "sawtooth"
  C  complex four-population model            <- demography == "complex_4_pops"
  D  two-group 2D stepping stone (50 demes)   <- demography == "double_2D_SS"
  Left strip: one time arrow per row. Right strip: per-row colour key.

Data: results/figure_data/sim_vs_singer/, written by
utils/runs/prepare_figure_data/sim_vs_singer.py. Nothing is computed here.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.patches
from matplotlib.lines import Line2D

from results.figures.plotting_style import *

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))
from PlotEmbeddings import PlotEmbeddings

FIGURE = "sim_vs_singer"
KIND = "app"

# ---------------------------------------------------------------------------
# Plotting parameters
# ---------------------------------------------------------------------------
WIDTH = "double"
ROW_LABEL_W = 0.22
TITLE_H = 0.22
ROW_GAP = 0.05
KEY_W = 1.40

DEMOGRAPHIES = ["split", "sawtooth", "complex_4_pops", "double_2D_SS"]
ROW_LETTERS = dict(zip(DEMOGRAPHIES, "ABCD"))
SOURCES = ["simulated trees", "SINGER trees"]
COLUMN_TITLES = {"simulated trees": "Simulated trees", "SINGER trees": "SINGER trees"}
GROUP_NAMES = {"g0": "Group A demes", "g1": "Group B demes"}

RENDER_KW = dict(width=900, height=900)
RENDER_SCALE = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    d = {}
    d["embeddings"] = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
    d["pop_labels"] = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
    d["pop_colors"] = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
    params = pd.read_csv(figure_data(FIGURE, "params.csv"))
    d["params"] = {dem: dict(zip(g["parameter"], g["value"]))
                   for dem, g in params.groupby("demography")}
    return d


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def render_embeddings(d, demography, source, scale=RENDER_SCALE):
    e = d["embeddings"]
    df = e[(e["demography"] == demography) & (e["source"] == source)]
    p = d["params"][demography]
    plotter = PlotEmbeddings(
        embeddings_df=df[["individual_id", "time", "X1", "X2"]].copy(),
        labels_df=d["pop_labels"][d["pop_labels"]["demography"] == demography],
        colors_df=d["pop_colors"][d["pop_colors"]["demography"] == demography],
        sample_id_col="individual_id")
    plotter.init_plot(min_time=float(p["min_time"]), max_time=float(p["max_time"]),
                      time_scale=p["time_scale"], xy_lims=float(p["xy_lims"]))
    plotter.add_sample_trajectories()
    return plotter.save_static(**RENDER_KW, scale=scale)


def key_panel(ax, d, demography):
    """Colour key for one row, in the right strip."""
    ax.set_axis_off()
    colors = d["pop_colors"][d["pop_colors"]["demography"] == demography]
    if demography == "double_2D_SS":
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        groups = colors["population"].str.split("_").str[0]
        for k, (group, name) in enumerate(GROUP_NAMES.items()):
            ramp = colors.loc[groups == group, "color"].tolist()
            y = 0.80 - 0.22 * k
            ax.text(0.04, y + 0.045, name, ha="left", va="bottom",
                    fontsize=FONT_SIZES["legend"])
            for i, c in enumerate(ramp):
                ax.add_patch(matplotlib.patches.Rectangle(
                    (0.04 + 0.72 * i / len(ramp), y - 0.03), 0.72 / len(ramp), 0.06,
                    facecolor=c, edgecolor="none", transform=ax.transAxes))
        return
    if len(colors) == 1:
        handles = [Line2D([], [], color=colors["color"].iloc[0], lw=1.5,
                          label="Single population")]
    else:
        colors = colors.sort_values("population")
        handles = [Line2D([], [], color=c, lw=1.5, label=f"Population {pop}")
                   for pop, c in zip(colors["population"], colors["color"])]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.88),
              handlelength=1.4, borderaxespad=0)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble(d):
    fig_w = WIDTHS[WIDTH]
    aspect = max(render_embeddings(d, dem, SOURCES[0], scale=1)[1] for dem in DEMOGRAPHIES)
    key_w = KEY_W
    panel_w = (fig_w - ROW_LABEL_W - key_w) / 2
    panel_h = panel_w / aspect
    n = len(DEMOGRAPHIES)
    heights = [TITLE_H]
    for i in range(n):
        heights += [panel_h] + ([ROW_GAP] if i < n - 1 else [])
    fig_h = sum(heights)
    fig = new_figure(WIDTH, fig_h)
    gs = inch_gridspec(fig, heights=heights, widths=[ROW_LABEL_W, panel_w, panel_w, key_w])
    row_index = [1 + 2 * i for i in range(n)]

    for r in row_index:
        strip_ax = fig.add_subplot(gs.cell(r, 0))
        strip_ax.set_axis_off()
        strip_ax.annotate("", xy=(0.80, 0.95), xytext=(0.80, 0.2), xycoords="axes fraction",
                          arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black",
                                          shrinkA=0, shrinkB=0, mutation_scale=7))
        strip_ax.text(0.40, 0.575, TIME_LABEL, rotation=90, ha="center", va="center",
                      fontsize=FONT_SIZES["annotation"], transform=strip_ax.transAxes)

    for i, demography in enumerate(DEMOGRAPHIES):
        r = row_index[i]
        for j, source in enumerate(SOURCES):
            ax = fig.add_subplot(gs.cell(r, 1 + j))
            img, _ = render_embeddings(d, demography, source)
            place_image(ax, img)
            if i == 0:
                ax.set_title(COLUMN_TITLES[source], fontsize=FONT_SIZES["title"], pad=2)
            if j == 0:
                add_panel_label(ax, ROW_LETTERS[demography], dx=-ROW_LABEL_W * 72, dy=0)
        key_panel(fig.add_subplot(gs.cell(r, 3)), d, demography)
    return fig


#%%
if __name__ == "__main__":
    save_figure(assemble(load_data()), FIGURE, KIND)
# %%
