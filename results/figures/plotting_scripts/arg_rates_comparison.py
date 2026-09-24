#%%
"""
Pairwise coalescent rates recovered from ARGs inferred by different methods, against
the analytic truth and the estimate on the true trees.

Panels
  A  population split, split in two stacked sub-panels sharing an x axis:
     within-population pairs (top) and between-population pairs (bottom); dashed
     verticals in the bottom sub-panel mark where each inferred-ARG source's
     between/within mean-rate ratio first reaches 0.5
                                       <- rates_split_<source>.parquet, mean_rates.parquet,
                                          ratio_half_times.csv
  B  single sawtooth population (within pairs only); bottom-left table of the
     fraction of each true log contrast between consecutive extrema that each
     inferred ARG retains, each curve read at its own extremum
                                       <- rates_sawtooth_<source>.parquet, mean_rates.parquet,
                                          contrast_retained.csv

Which rows of a rates_*.parquet are within- vs between-population pairs comes from
pairs.csv (one row per pair, in the same order as the parquet rows).

Data: results/figure_data/arg_rates_comparison/, written by
utils/runs/prepare_figure_data/arg_rates_comparison.py. Every source is already on
one time grid per demography, so this script only draws.
"""
import os

import matplotlib.patches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from results.figures.plotting_style import *

FIGURE = os.environ.get("ARG_RATES_FIGURE_DATA", "arg_rates_comparison")
KIND = "app"

# ---------------------------------------------------------------------------
# Plotting parameters: the only knobs. Inches unless stated.
# ---------------------------------------------------------------------------
WIDTH = "double"
MARGIN_L = 0.64         # y-axis label + tick labels
MARGIN_B = 0.38         # x-axis label + tick labels
MARGIN_T = 0.16         # panel letters sit above the frame
MARGIN_R = 0.06
COL_GAP = 0.02          # between B's y-axis label strip and column A
ROW_GAP = 0.16          # between A's two stacked sub-panels (they share the x axis)
LEGEND_H = 0.30         # strip for the shared legend
# panel B is square, so the column width sets the height of both columns; A's two
# sub-panels split that height equally, so they match each other and B

# analytic truth in black, the estimate on the true trees in grey, one Okabe-Ito
# colour per inference method
SOURCE_COLORS = {
    "true": TRUTH_COLOR,
    "sim_trees": SIMULATED_COLOR,
    "tsinfer": okabe_ito[0],
    "relate": okabe_ito[1],
    "singer": okabe_ito[2],
}
SOURCE_LABELS = {
    "true": "analytic truth",
    "sim_trees": "true trees",
    "tsinfer": "tsinfer + tsdate",
    "relate": "Relate",
    "singer": "SINGER + POLEGON",
}
# between-population rates fall to the 1e-16 clip before the split, so their axis is
# floored well above it. Within-population rates never leave a factor of ~3 around
# 1e-4, so their sub-panel is floored a decade below that instead of spending three
# decades on empty space; the floor is a round 1e-5 so the axis carries decade ticks.
RATE_FLOOR = 1e-6
RATE_FLOOR_WITHIN_SPLIT = 1e-5
CLOUD_KW = dict(lw=0.15, alpha=0.08, zorder=1, rasterized=True)
# the means overlap heavily wherever the methods agree, and an opaque line simply
# hides whichever source was drawn before it: draw them translucent so overlaps
# blend and the truth stays visible under the estimates
MEAN_KW = dict(lw=1.6, alpha=0.75, zorder=3)
HALF_TIME_KW = dict(ls="--", lw=1.0, zorder=4)

# the three (demography, pair_type) cells: y label and axis floor for each
CELLS = {
    ("split", "within"): ("Within-population\ncoalescent rates", RATE_FLOOR_WITHIN_SPLIT),
    ("split", "between"): ("Between-population\ncoalescent rates", RATE_FLOOR),
    ("sawtooth", "within"): ("Coalescent rates", RATE_FLOOR),
}


# ---------------------------------------------------------------------------
# Data: read, index, nothing else.
# ---------------------------------------------------------------------------
def load_data():
    d = {"rates": {}}
    d["mean_rates"] = pd.read_parquet(figure_data(FIGURE, "mean_rates.parquet"))
    d["pairs"] = pd.read_csv(figure_data(FIGURE, "pairs.csv"))
    d["params"] = pd.read_csv(figure_data(FIGURE, "params.csv"))
    d["half_times"] = pd.read_csv(figure_data(FIGURE, "ratio_half_times.csv"))
    d["contrast"] = pd.read_csv(figure_data(FIGURE, "contrast_retained.csv"))
    for demography in ("split", "sawtooth"):
        for source in SOURCE_COLORS:
            d["rates"][demography, source] = pd.read_parquet(
                figure_data(FIGURE, f"rates_{demography}_{source}.parquet"))
    return d


def param(d, demography, name, cast=float):
    params = d["params"]
    value = params.loc[(params["demography"] == demography)
                       & (params["parameter"] == name), "value"].iloc[0]
    return cast(value)


# ---------------------------------------------------------------------------
# Panels: one cell = one demography x pair_type combination.
# ---------------------------------------------------------------------------
def panel_rates(ax, d, demography, pair_type, xlabel=True):
    """Every pair as a thin translucent line, the per-source mean on top."""
    pairs = d["pairs"][d["pairs"]["demography"] == demography]
    keep = (pairs["pair_type"] == pair_type).to_numpy()

    # all sources' clouds first, so no source's cloud is drawn over another's mean
    for source, color in SOURCE_COLORS.items():
        rates = d["rates"][demography, source]
        times = rates.columns.values.astype(float)
        for row in rates.to_numpy()[keep]:
            ax.plot(times, row, color=color, **CLOUD_KW)

    means = d["mean_rates"]
    means = means[(means["demography"] == demography)
                  & (means["pair_type"] == pair_type)]
    for source, color in SOURCE_COLORS.items():
        group_mean = means[means["source"] == source]
        if group_mean.empty:
            continue
        ax.plot(group_mean["time"], group_mean["rate"], color=color, **MEAN_KW)

    if pair_type == "between":
        half = d["half_times"][d["half_times"]["demography"] == demography]
        for source, time in zip(half["source"], half["time"]):
            ax.axvline(time, color=SOURCE_COLORS[source], **HALF_TIME_KW)

    ylabel, floor = CELLS[demography, pair_type]
    ax.set_xscale(param(d, demography, "time_scale", str))
    log_axis(ax, "y")
    # The config keys these come from were renamed (min_time -> embeddings_min_time,
    # max_time -> pdf_max_time); params.csv carries the config's own key names.
    ax.set_xlim(param(d, demography, "embeddings_min_time"),
                param(d, demography, "pdf_max_time"))
    ax.set_ylim(floor, param(d, demography, "max_rate_plot"))
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(TIME_LABEL)
    else:
        ax.tick_params(labelbottom=False)


TABLE_LABELS = {"tsinfer": "tsinfer", "relate": "Relate", "singer": "SINGER+POLEGON"}


def contrast_table(ax, d):
    """rho per inferred-ARG source (rows) and pair of consecutive extrema (columns)."""
    c = d["contrast"]
    pairs = c.drop_duplicates("contrast").sort_values("contrast")
    x_label, x_cols, y_top, dy = 0.04, [0.48, 0.665, 0.86], 0.235, 0.048
    kw = dict(transform=ax.transAxes, fontsize=FONT_SIZES["annotation"], zorder=6)
    ax.add_patch(matplotlib.patches.Rectangle(
        (0.015, 0.015), 0.95, y_top + 0.03, transform=ax.transAxes, facecolor="white",
        edgecolor="none", alpha=0.85, zorder=5))
    ax.text((x_cols[0] + x_cols[-1]) / 2, y_top, r"Log contrast retained, $\rho$", ha="center", va="center", **kw)
    for x, (_, pair) in zip(x_cols, pairs.iterrows()):
        ax.text(x, y_top - dy, f"{pair.time_a:,.0f}\u2013{pair.time_b:,.0f}",
                ha="center", va="center", **kw)
    for i, source in enumerate(c["source"].drop_duplicates(), start=2):
        y = y_top - i * dy
        ax.text(x_label, y, TABLE_LABELS[source], ha="left", va="center",
                color=SOURCE_COLORS[source], **kw)
        rows = c[c["source"] == source].sort_values("contrast")
        for x, rho in zip(x_cols, rows["rho"]):
            ax.text(x, y, f"{rho:.2f}", ha="center", va="center", **kw)


# ---------------------------------------------------------------------------
# Assembly: owns every inch of the figure.
# ---------------------------------------------------------------------------
def assemble(d):
    fig_w = WIDTHS[WIDTH]
    panel_w = (fig_w - 2 * MARGIN_L - MARGIN_R - COL_GAP) / 2
    panel_h = panel_w                      # panel B is square
    fig_h = MARGIN_T + panel_h + MARGIN_B + LEGEND_H
    fig = new_figure(WIDTH, fig_h)

    gs = inch_gridspec(fig,
                       heights=[MARGIN_T, panel_h, MARGIN_B, LEGEND_H],
                       widths=[MARGIN_L, panel_w, COL_GAP, MARGIN_L, panel_w, MARGIN_R])

    # column A is the same strip split into two sub-panels of equal height that
    # share the x axis; only the bottom one keeps tick labels, and the x label
    # lives in the shared MARGIN_B strip below, so neither sub-panel pays for it
    sub_h = (panel_h - ROW_GAP) / 2
    left = inch_gridspec(gs.cell(1, 1), heights=[sub_h, sub_h], hgap=ROW_GAP)
    ax_within = fig.add_subplot(left.cell(0))
    ax_between = fig.add_subplot(left.cell(1), sharex=ax_within)
    ax_saw = fig.add_subplot(gs.cell(1, 4))

    panel_rates(ax_within, d, "split", "within", xlabel=False)
    panel_rates(ax_between, d, "split", "between")
    panel_rates(ax_saw, d, "sawtooth", "within")
    contrast_table(ax_saw, d)

    # one letter per logical panel: A labels the stacked pair, B the sawtooth
    add_panel_label(ax_within, "A", dx=2, dy=13)
    add_panel_label(ax_saw, "B", dx=2, dy=13)

    handles = [Line2D([], [], color=c, lw=1.8, label=SOURCE_LABELS[s])
               for s, c in SOURCE_COLORS.items()]
    fig.legend(handles=handles, loc="center", ncol=len(handles),
               bbox_to_anchor=(0.5, LEGEND_H / 2 / fig_h))
    return fig


#%%
if __name__ == "__main__":
    save_figure(assemble(load_data()), FIGURE, KIND)
# %%
