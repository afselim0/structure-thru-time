#%%
"""
The two-group 2D stepping-stone example on simulated versus SINGER-inferred trees:
the two embeddings through time (row 1), and the mean pairwise coalescent rates for
the neighbouring-deme pairs of each grid and for the three coarse pair classes
(row 2).

Data: results/figure_data/double_SS_2d_singer/, written by
utils/runs/prepare_figure_data/double_SS_2d_singer.py -- the SINGER embeddings there
are already Procrustes-aligned to the simulated-tree ones and the rate trajectories
are already averaged over the individual pairs of each class.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
filename = os.path.basename(__file__).split(".")[0]
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
from PlotEmbeddings import *

FIGURE = "double_SS_2d_singer"

# the tree sources, and the line style each is drawn with -- the same convention as
# the other figures in this SINGER comparison set
SOURCE_STYLES = {"simulated trees": "--", "SINGER trees": ":"}
EMBEDDING_SOURCES = ["simulated trees", "SINGER trees"]
COLUMN_TITLES = {"simulated trees": "Simulated trees", "SINGER trees": "SINGER trees"}
PANEL_LETTERS = ["A", "B", "C", "D"]

# the pair classes, coloured as in plotting_scripts/double_SS_2d.py so the two
# figures of this example read the same way
PAIR_COLORS = {"g0": "tab:orange", "g1": "tab:green", "between": "tab:brown"}
PAIR_NAMES = {"g0": "Group A", "g1": "Group B", "between": "Between groups"}

embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
mean_rates = pd.read_parquet(figure_data(FIGURE, "mean_rates.parquet"))
pop_labels = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
pop_colors = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
params = pd.read_csv(figure_data(FIGURE, "params.csv")).set_index("parameter")["value"]

min_time = float(params["min_time"])
max_time = float(params["max_time"])
time_scale = params["time_scale"]
xy_lims = float(params["embeddings_plot_lims"])
max_rate_plot = float(params["max_rate_plot"])
rate_floor = float(params["rate_floor"])

# recolour demes by their "g<N>_" group, exactly as plotting_scripts/double_SS_2d.py
# does: each group gets its own sequential colormap, so the two groups stay visually
# distinct while the ramp within a group tracks deme order
GROUP_CMAPS = {"g0": "spring", "g1": "winter"}
CMAP_RANGE = (0.15, 0.95)   # skip the pale low end so no deme is near-white

pop_groups = pop_colors["population"].str.split("_").str[0]
for group, cmap_name in GROUP_CMAPS.items():
    members = pop_groups == group
    cmap = plt.get_cmap(cmap_name)
    shades = np.linspace(*CMAP_RANGE, members.sum())
    pop_colors.loc[members, "color"] = [mcolors.to_hex(cmap(s)[:3]) for s in shades]


def rate_line(ax, source, panel, pair_class):
    """Draw one (source, pair class) trajectory, floored for the log axis."""
    selected = mean_rates[(mean_rates["source"] == source)
                          & (mean_rates["panel"] == panel)
                          & (mean_rates["pair_class"] == pair_class)]
    selected = selected.sort_values("time")
    ax.plot(selected["time"], np.maximum(selected["rate"], rate_floor),
            color=PAIR_COLORS[pair_class], linestyle=SOURCE_STYLES[source], lw=1.1)


# the panel is rendered on this canvas; the cropped aspect ratio depends on the
# canvas size, so the probe and the final render must share it
RENDER_KW = dict(width=1200, height=1200)


def render_panel(source, scale=4):
    """Render one static 3D embeddings panel and return (image, aspect ratio)."""
    df = embeddings[embeddings["source"] == source]
    plotter = PlotEmbeddings(
        embeddings_df=df[["individual_id", "time", "X1", "X2"]].copy(),
        labels_df=pop_labels, colors_df=pop_colors, sample_id_col="individual_id")
    plotter.init_plot(min_time=min_time, max_time=max_time,
                      time_scale=time_scale, xy_lims=xy_lims)
    plotter.add_sample_trajectories()
    return plotter.save_static(**RENDER_KW, scale=scale)


#%%
# ---- layout: the embedding row's height follows its render, the rate row is fixed ----
_, aspect_ratio = render_panel(EMBEDDING_SOURCES[0], scale=1)

FIG_W = 6.5
EMB_H = FIG_W / 2 / aspect_ratio    # two embedding panels spanning the full width
TITLE_H = 0.28                      # strip above row 1 holding the column titles
RATE_H = 2.1                        # height of the rate row
FIG_H = EMB_H + TITLE_H + RATE_H

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=500)
outer = gridspec.GridSpec(
    nrows=3, ncols=1, height_ratios=[TITLE_H, EMB_H, RATE_H], hspace=0,
    left=0, right=1, top=1, bottom=0, figure=fig)
emb_grid = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=2, subplot_spec=outer[1], wspace=0)
# the rate panels carry tick labels, an x label and a legend, none of which a nested
# GridSpec makes room for; a 3x3 frame around them supplies the margins in inches,
# and the panels themselves sit in its centre cell
RATE_MARGIN = dict(left=0.62, right=0.10, top=0.30, bottom=0.60)
rate_frame = gridspec.GridSpecFromSubplotSpec(
    nrows=3, ncols=3, subplot_spec=outer[2], wspace=0, hspace=0,
    width_ratios=[RATE_MARGIN["left"],
                  FIG_W - RATE_MARGIN["left"] - RATE_MARGIN["right"],
                  RATE_MARGIN["right"]],
    height_ratios=[RATE_MARGIN["top"],
                   RATE_H - RATE_MARGIN["top"] - RATE_MARGIN["bottom"],
                   RATE_MARGIN["bottom"]])
rate_grid = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=2, subplot_spec=rate_frame[1, 1], wspace=0.30)

emb_axes = [fig.add_subplot(emb_grid[0, j]) for j in range(2)]
neighbour_ax = fig.add_subplot(rate_grid[0, 0])
all_ax = fig.add_subplot(rate_grid[0, 1])

#%%
# ---- row 1: the two embeddings ----
for j, source in enumerate(EMBEDDING_SOURCES):
    ax = emb_axes[j]
    img, _ = render_panel(source)
    img.save(f"../app_figs_png/{filename}_{source.split()[0]}.png")
    ax.set_axis_off()
    ax.imshow(img, aspect="equal")
    ax.text(-0.02, 1.02, PANEL_LETTERS[j], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")
    pos = ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.005, COLUMN_TITLES[source],
             ha="center", va="bottom", fontsize=11)

# ---- row 2: mean rates, colour for the pair class, line style for the source ----
# left panel: only the lattice-neighbour pairs of each grid; right panel: every pair
# within a grid, and every pair spanning the two
PANEL_CLASSES = {"neighbour": ["g0", "g1"], "all": ["g0", "g1", "between"]}
for source in SOURCE_STYLES:
    for panel, ax in (("neighbour", neighbour_ax), ("all", all_ax)):
        for pair_class in PANEL_CLASSES[panel]:
            rate_line(ax, source, panel, pair_class)

# the two panels span very different rate ranges -- the neighbouring-deme rates sit
# in one decade while the between-group ones fall four below -- so each panel takes
# the decades its own trajectories occupy rather than a shared range that would
# flatten the left panel into a single line
def panel_ylim(panel):
    values = mean_rates.loc[mean_rates["panel"] == panel, "rate"]
    values = np.maximum(values, rate_floor)
    lo = 10 ** np.floor(np.log10(values.min()))
    hi = min(10 ** np.ceil(np.log10(values.max())), max_rate_plot)
    return lo, hi


for ax, letter, title, panel in (
        (neighbour_ax, PANEL_LETTERS[2], "Neighbouring demes", "neighbour"),
        (all_ax, PANEL_LETTERS[3], "All pairs", "all")):
    ax.set_xscale(time_scale)
    ax.set_yscale("log")
    ax.set_xlim(min_time, max_time)
    ax.set_ylim(*panel_ylim(panel))
    ax.set_xlabel("Time (generations ago)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7.5)
    ax.text(-0.20, 1.12, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")
neighbour_ax.set_ylabel("Mean coalescent rate", fontsize=9)

# two keys per panel: which pair class a colour is, and which source a line style is
neighbour_ax.legend(
    handles=[Line2D([], [], color=PAIR_COLORS[c], lw=1.6,
                    label=f"{PAIR_NAMES[c]}, neighbouring demes")
             for c in PANEL_CLASSES["neighbour"]],
    loc="lower left", fontsize=6.5, handlelength=1.2, handletextpad=0.4,
    frameon=False)
# the right panel is busy top (the two within-group plateaus) and bottom right (the
# climbing between-group line), leaving its middle-left corner as the only clear space
all_ax.legend(
    handles=[Line2D([], [], color=PAIR_COLORS[c], lw=1.6, label=PAIR_NAMES[c])
             for c in PANEL_CLASSES["all"]],
    loc="center left", fontsize=6.5, handlelength=1.2, handletextpad=0.4,
    frameon=False)
fig.legend(
    handles=[Line2D([], [], color="black", linestyle=style, lw=1.2, label=source)
             for source, style in SOURCE_STYLES.items()],
    loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8,
    handlelength=2.2, frameon=False)

fig.savefig(f"../app_figs/{filename}.pdf", bbox_inches="tight", pad_inches=0.02)
# the same figure as a PNG, for quick viewing without a PDF reader
fig.savefig(f"../app_figs_png/{filename}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print(f"Saved ../app_figs/{filename}.pdf")
# %%
