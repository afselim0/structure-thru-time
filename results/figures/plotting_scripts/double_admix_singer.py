#%%
"""
The two-admixture example on SINGER-inferred trees, under GNN and RFmix local
ancestry. Columns are the four source ancestries A-D.

Panels
  A  per-ancestry 3D embeddings, GNN LAI            <- embeddings.parquet (method == "GNN")
  B  admixed-vs-everyone coalescence rates, GNN     <- input_rates.parquet (level method == "GNN")
  C  as A, RFmix LAI                                <- embeddings.parquet (method == "RFmix")
  D  as B, RFmix LAI                                <- input_rates.parquet (level method == "RFmix")
  Shared population legend below; one rate scale (rate_limits.csv) for B and D.

Embeddings are aligned onto the simulated-tree GNN frames of
sim_figs/double_admix_GNN.pdf.

Data: results/figure_data/double_admix_singer/, written by
utils/runs/prepare_figure_data/double_admix_singer.py. Nothing is computed here.
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from results.figures.plotting_style import *

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))
from PlotEmbeddings import PlotEmbeddings

FIGURE = "double_admix_singer"
KIND = "app"

# ---------------------------------------------------------------------------
# Plotting parameters: the only knobs. Inches unless stated.
# ---------------------------------------------------------------------------
WIDTH = "double"
BLOCK_TITLE_W = 0.24   # rotated "GNN-based LAI" / "RFmix-based LAI"
Y_AXIS_W = 0.55        # rate y label + ticks; holds the time arrow in embedding rows
RIGHT_W = 0.06
COL_GAP = 0.08
COL_TITLE_H = 0.24
RATE_H = 1.05
XLABEL_H = 0.40
BLOCK_GAP = 0.14
LEGEND_H = 0.32

ANCESTRIES = ["A", "B", "C", "D"]
METHODS = ["GNN", "RFmix"]
BLOCK_TITLES = {"GNN": "GNN-based LAI", "RFmix": "RFmix-based LAI"}
LETTERS = {"GNN": ("A", "B"), "RFmix": ("C", "D")}
RATE_LABEL = r"$\hat{\lambda}_{ij}(t) + 1/N_{\max}$"

RENDER_KW = dict(width=700, height=700)
RENDER_SCALE = 3
TIME_TICK_PX = 16
INDIVIDUAL_STYLE = dict(lw=0.2, alpha=0.4, rasterized=True)
MEAN_STYLE = dict(lw=1.2, alpha=0.9)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    d = {}
    d["pop_labels"] = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
    d["pop_colors"] = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
    d["color_by_pop"] = dict(zip(d["pop_colors"]["population"], d["pop_colors"]["color"]))
    d["admix_pop"] = (pd.read_csv(figure_data(FIGURE, "admix_populations.csv"))
                      .set_index("ancestry")["admix_pop"].to_dict())
    lims = pd.read_csv(figure_data(FIGURE, "rate_limits.csv")).iloc[0]
    d["rate_lims"] = (float(lims["rate_lo"]), float(lims["rate_hi"]))
    params = pd.read_csv(figure_data(FIGURE, "params.csv"))
    d["params"] = dict(zip(params["parameter"], params["value"]))
    emb = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
    rates = pd.read_parquet(figure_data(FIGURE, "input_rates.parquet"))
    d["embeddings"] = {(m, a): emb[(emb["method"] == m) & (emb["ancestry"] == a)]
                       for m in METHODS for a in ANCESTRIES}
    d["rates"] = {(m, a): rates.xs((m, a), level=["method", "ancestry"])
                  for m in METHODS for a in ANCESTRIES}
    return d


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def render_embeddings(d, method, ancestry, scale=RENDER_SCALE):
    p = d["params"]
    plotter = PlotEmbeddings(embeddings_df=d["embeddings"][(method, ancestry)],
                             labels_df=d["pop_labels"], colors_df=d["pop_colors"],
                             sample_id_col="individual_id")
    plotter.init_plot(min_time=float(p["min_time"]), max_time=float(p["max_time"]),
                      time_scale=p["time_scale"], xy_lims=float(p["xy_lims"]))
    plotter.add_sample_trajectories()
    hidden = dict(showticklabels=False, title="")
    # time ticks only in the first column; they widen that render on the left only,
    # so every image is anchored right in its cell and the panes stay aligned
    zaxis = (dict(title="", tickfont=dict(size=TIME_TICK_PX))
             if ancestry == ANCESTRIES[0] else hidden)
    plotter.fig.update_layout(scene=dict(xaxis=hidden, yaxis=hidden, zaxis=zaxis))
    return plotter.save_static(**RENDER_KW, scale=scale)


def panel_embeddings(ax, d, method, ancestry):
    img, _ = render_embeddings(d, method, ancestry)
    place_image(ax, img)
    ax.set_anchor("E")
    ax.set_anchor("E")


def panel_rates(ax, d, method, ancestry):
    """Rates between this column's admixed population and every population, thin per pair, thick mean."""
    rates = d["rates"][(method, ancestry)]
    labels = d["pop_labels"]
    admix_pop = d["admix_pop"][ancestry]
    N_max = float(d["params"]["N_max"])
    active = np.unique([s for pair in rates.index.values for s in pair])
    labels = labels[labels["individual_id"].isin(active)]
    times = rates.columns.astype(float)

    admix_samples = np.unique(labels.loc[labels["population"] == admix_pop, "individual_id"])
    partners = [p for p in labels["population"].unique() if p != admix_pop] + [admix_pop]
    for partner in partners:
        partner_samples = np.unique(labels.loc[labels["population"] == partner, "individual_id"])
        pairs = [(a, b) if a < b else (b, a)
                 for a in admix_samples for b in partner_samples if a != b]
        pairs = [pr for pr in dict.fromkeys(pairs) if pr in rates.index]
        if not pairs:
            continue
        values = rates.loc[pairs, :].values + 1 / N_max
        color = d["color_by_pop"][partner]
        for row in values:
            ax.plot(times, row, color=color, **INDIVIDUAL_STYLE)
        ax.plot(times, values.mean(axis=0), color=color, **MEAN_STYLE)

    log_axis(ax, "x")
    log_axis(ax, "y")
    ax.set_xlim(float(d["params"]["min_time"]), float(d["params"]["max_time"]))
    ax.set_ylim(d["rate_lims"])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def assemble(d):
    fig_w = WIDTHS[WIDTH]
    _, aspect = render_embeddings(d, METHODS[0], ANCESTRIES[0], scale=1)
    panel_w = (fig_w - BLOCK_TITLE_W - Y_AXIS_W - RIGHT_W - 3 * COL_GAP) / 4
    emb_h = panel_w / aspect

    heights = [COL_TITLE_H]
    rows = {}
    for k, method in enumerate(METHODS):
        if k:
            heights.append(BLOCK_GAP)
        rows[method] = len(heights)
        heights += [emb_h, RATE_H, XLABEL_H]
    heights.append(LEGEND_H)
    fig_h = sum(heights)
    fig = new_figure(WIDTH, fig_h)

    widths = [BLOCK_TITLE_W, Y_AXIS_W]
    for j in range(4):
        if j:
            widths.append(COL_GAP)
        widths.append(panel_w)
    widths.append(RIGHT_W)
    gs = inch_gridspec(fig, heights=heights, widths=widths)
    col = {a: 2 + 2 * j for j, a in enumerate(ANCESTRIES)}  # gs.cell column index

    for method in METHODS:
        r = rows[method]
        emb_letter, rate_letter = LETTERS[method]

        title_ax = fig.add_subplot(gs[2 * r:2 * r + 5, 0])
        title_ax.set_axis_off()
        title_ax.text(0.35, 0.5, BLOCK_TITLES[method], rotation=90, ha="center", va="center",
                      fontsize=FONT_SIZES["title"], transform=title_ax.transAxes)

        arrow_ax = fig.add_subplot(gs.cell(r, 1))
        arrow_ax.set_axis_off()
        arrow_ax.annotate("", xy=(0.85, 0.95), xytext=(0.85, 0.2), xycoords="axes fraction",
                          arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black",
                                          shrinkA=0, shrinkB=0, mutation_scale=7))
        arrow_ax.text(0.62, 0.575, TIME_LABEL, rotation=90, ha="center", va="center",
                      fontsize=FONT_SIZES["annotation"], transform=arrow_ax.transAxes)

        first_rate_ax = None
        for j, ancestry in enumerate(ANCESTRIES):
            emb_ax = fig.add_subplot(gs.cell(r, col[ancestry]))
            panel_embeddings(emb_ax, d, method, ancestry)
            if method == METHODS[0]:
                emb_ax.set_title(f"{ancestry} ancestry", fontsize=FONT_SIZES["title"], pad=2)
            if j == 0:
                add_panel_label(emb_ax, emb_letter, dx=-42, dy=0)

            rate_ax = fig.add_subplot(gs.cell(r + 1, col[ancestry]),
                                      sharex=first_rate_ax, sharey=first_rate_ax)
            panel_rates(rate_ax, d, method, ancestry)
            if j == 0:
                first_rate_ax = rate_ax
                rate_ax.set_ylabel(RATE_LABEL)
                add_panel_label(rate_ax, rate_letter, dx=-42, dy=10)
            else:
                rate_ax.tick_params(labelleft=False)

        xlabel_ax = fig.add_subplot(gs[2 * (r + 2), 2 * col[ANCESTRIES[0]]:])
        xlabel_ax.set_axis_off()
        xlabel_ax.text(0.5, 0.05, TIME_LABEL, ha="center", va="bottom",
                       transform=xlabel_ax.transAxes)

    handles = [Line2D([], [], color=c, linewidth=1.5, label=p)
               for p, c in d["color_by_pop"].items()]
    grid_x0 = BLOCK_TITLE_W + Y_AXIS_W
    fig.legend(handles=handles, loc="center", ncol=len(handles),
               bbox_to_anchor=((grid_x0 + (fig_w - grid_x0) / 2) / fig_w, LEGEND_H / 2 / fig_h),
               handlelength=1.8, columnspacing=1.5)
    return fig


#%%
if __name__ == "__main__":
    save_figure(assemble(load_data()), FIGURE, KIND)
# %%
