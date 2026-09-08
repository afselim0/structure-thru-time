#%%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.legend_handler import HandlerPatch
import numpy as np
import pandas as pd
from results.figures.plotting_style import *
import sys
import yaml
import os
filename = os.path.basename(__file__).split(".")[0]
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils/snakemake_utils")
sys.path.append("../../../utils/misc_utils")
from PlotEmbeddings import *
config_file = "../../../data/configs/sim_configs/double_2d_SS.yaml"
N_max = 1e6
embeddings_df = pd.read_parquet("../../outputs/simulations/double_2d_SS/sim/data/smooth_embeddings.d2.Nmin1.Nmax10000000.p2.linkinverse_distance.L1[1e-2,1e-3].nTe2000.minWS50.maxWS200.V1.FS1.0.maxT6000.nT2000.parquet")
estimated_rates = pd.read_parquet("../../outputs/simulations/double_2d_SS/sim/data/rates.minWS50.maxWS200.V1.FS1.0.maxT6000.nT2000.parquet") + 1/N_max
reconstructed_rates = pd.read_parquet("../../outputs/simulations/double_2d_SS/sim/data/smooth_reconstructed_rates.d2.Nmin1.Nmax10000000.p2.linkinverse_distance.L1[1e-2,1e-3].nTe2000.minWS50.maxWS200.V1.FS1.0.maxT6000.nT2000.parquet") + 1/N_max
pop_labels = pd.read_csv("../../outputs/simulations/double_2d_SS/sim/data/pop_labels.csv")
pop_colors = pd.read_csv("../../outputs/simulations/double_2d_SS/sim/data/pop_colors.csv")
times = estimated_rates.columns.astype(float)
min_time = 50
# recolor demes by their "g<N>_" group: each group gets its own sequential colormap, so the
# two groups stay visually distinct while the ramp within a group tracks deme order
GROUP_CMAPS = {"g0": "spring", "g1": "winter"}
CMAP_RANGE = (0.15, 0.95)  # skip the pale low end so no deme is near-white

pop_groups = pop_colors["population"].str.split("_").str[0]
for group, cmap_name in GROUP_CMAPS.items():
    members = pop_groups == group
    cmap = plt.get_cmap(cmap_name)
    shades = np.linspace(*CMAP_RANGE, members.sum())
    pop_colors.loc[members, "color"] = [mcolors.to_hex(cmap(s)[:3]) for s in shades]
t1 = 150
t2 = 1500
t3 = 4500
t4 = 5400
plotter = PlotEmbeddings(embeddings_df = embeddings_df, labels_df = pop_labels, colors_df = pop_colors, sample_id_col = "individual_id")
plotter.init_plot(xy_lims=2000, min_time = min_time)
plotter.add_sample_trajectories()

# the embeddings panel's width follows its true rendered aspect ratio, so it fills its
# axis without whitespace or distortion; the schematic takes whatever width is left
# the panel is TOP_H inches tall at 500 dpi; the cropped aspect ratio depends on this
# canvas size, so the probe here and the final render below must share it
RENDER_KW = dict(width=1500, height=1500)
_, aspect_ratio = plotter.save_static(**RENDER_KW, scale=1)  # cheap probe: scale does not affect the ratio
# %% figure layout
FIG_W, FIG_H = 6.5, 6.5
TOP_H = 3.0                      # height of the schematic / embeddings row
BOTTOM_H = 3.5                   # height of the 8-subplot region
PANEL = 1.5                      # each bottom subplot is 1.5 x 1.5

embeddings_w = TOP_H * aspect_ratio
schematic_w = FIG_W - embeddings_w

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=500)
# margins pinned to the full figure so the ratios below map to real inches
outer = gridspec.GridSpec(
    nrows=2, ncols=1, height_ratios=[TOP_H, BOTTOM_H], hspace=0,
    left=0, right=1, top=1, bottom=0, figure=fig,
)

# ---- top row: schematic of the two 5x5 grids (left) + static 3D embeddings (right) ----
top = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=2, subplot_spec=outer[0],
    width_ratios=[schematic_w, embeddings_w], wspace=0,
)
schematic_ax = fig.add_subplot(top[0, 0])
embeddings_ax = fig.add_subplot(top[0, 1])

# ---- bottom: 8 panels in 2 rows of 4, grouped into pairs ----
# placed by explicit inch coordinates rather than a GridSpec: gridspec spreads its
# wspace evenly between all columns, which cannot give the two gap sizes below
# while also holding every panel at the same size
# the gaps are set first and the panel width follows, so the row still spans the full
# figure width; the between-pair gap stays twice the within-pair one to show the pairing
within_pair_gap = 0.30
between_pair_gap = 0.45
panel_w = (FIG_W - 2 * within_pair_gap - between_pair_gap) / 4
row_gap = BOTTOM_H - 2 * PANEL       # spare height between the two rows

# x offset of each panel, walking left to right across the two pairs
panel_x = [
    0.0,
    panel_w + within_pair_gap,
    2 * panel_w + within_pair_gap + between_pair_gap,
    3 * panel_w + 2 * within_pair_gap + between_pair_gap,
]
# y offset of each row, measured from the bottom of the figure; the top row is dropped
# by TOP_ROW_DROP to open up space between it and the panels above
TOP_ROW_DROP = 0.20
panel_y = [BOTTOM_H - PANEL - TOP_ROW_DROP, 0.0]

bottom_axes = [
    [
        fig.add_axes([x / FIG_W, y / FIG_H, panel_w / FIG_W, PANEL / FIG_H])
        for x in panel_x
    ]
    for y in panel_y
]
#%%

# ---- bottom panels: each pair is (2D embedding slice, rate fit) at one timepoint ----
panel_times = [t1, t2, t3, t4]

# tick styling for the 1.5in panels, overriding the house style's full-page sizes
TICK_LABEL_SIZE = 6
TICK_LENGTH = 2
AXIS_LABEL_SIZE = 7   # the house style's 11pt is too large at this size

# inset placement in each embedding panel, as [x0, y0, w, h] in axes coords. g0's cluster
# sits upper-left and g1's lower-right, so each inset goes in the opposite free corner;
# once merged, the single cluster is central and its inset goes top-right.
INSET_RECTS = {
    "g0": [0.56, 0.60, 0.42, 0.38],
    "g1": [0.02, 0.02, 0.42, 0.38],
    "merged": [0.56, 0.60, 0.42, 0.38],
}
# which of the four connector lines to keep. matplotlib orders them
# (lower_left, upper_left, lower_right, upper_right), so the left pair is (0, 1) and the
# right pair is (2, 3). Only the two corners facing the inset should be drawn.
INSET_CORNERS = {
    "g0": (2, 3),        # inset sits to the right: use the right corners
    "g1": (0, 1),        # inset sits to the left: use the left corners
    "merged": (2, 3),    # single inset on the right: use the right corners
}
INSET_PAD_FRAC = 0.10   # padding around a cluster, as a fraction of its larger extent
INSET_GROUP_KEYS = ["g0", "g1"]   # deme-name prefixes, one cluster each until they merge
X_AXIS_LABEL_PAD = 1.0   # sit the label close to the tick labels
Y_LABEL_X = -0.12        # X2's x position in axes coords; less negative sits closer

# per-individual colors, so the 2D slices match the 3D trajectory panel
color_by_pop = dict(zip(pop_colors["population"], pop_colors["color"]))
individual_colors = (
    pop_labels.drop_duplicates("individual_id")
    .set_index("individual_id")["population"]
    .map(color_by_pop)
)

# rates are stored wide, one column per timepoint; the two frames label their columns
# with different dtypes (str vs float), so they are indexed positionally below
rate_times = estimated_rates.columns.astype(float)
if not np.allclose(rate_times, reconstructed_rates.columns.astype(float)):
    raise ValueError("Estimated and reconstructed rates are on different time grids")
if not estimated_rates.index.equals(reconstructed_rates.index):
    raise ValueError("Estimated and reconstructed rates cover different sample pairs")
embedding_times = np.sort(embeddings_df["time"].unique())

def nearest_rate_idx(t):
    """Positional index of the rates timepoint closest to t (grids need not coincide)."""
    return int(np.argmin(np.abs(rate_times - t)))

def nearest_embedding_time(t):
    """Embedding timepoint closest to t; the stored grid is coarser than the labels."""
    return embedding_times[np.argmin(np.abs(embedding_times - t))]

# color each pair by which groups its two individuals come from: within-g0, within-g1,
# or between the two groups -- so the rate panels show how the fit differs by pair type.
# chosen outside the red/blue/purple already used for the groups and their migration
# rates elsewhere in the figure, so the two color codings don't get conflated
PAIR_COLORS = {"g0": "tab:orange", "g1": "tab:green", "between": "tab:brown"}

individual_groups = (
    pop_labels.drop_duplicates("individual_id")
    .set_index("individual_id")["population"]
    .str.split("_").str[0]
)
group_1 = individual_groups.reindex(estimated_rates.index.get_level_values(0)).to_numpy()
group_2 = individual_groups.reindex(estimated_rates.index.get_level_values(1)).to_numpy()
if pd.isna(group_1).any() or pd.isna(group_2).any():
    raise ValueError("Some rate pairs reference individuals missing from pop_labels")
pair_types = np.where(group_1 == group_2, group_1, "between")

# shared limits so panels are comparable across timepoints
emb_lim = np.abs(embeddings_df[["X1", "X2"]].values).max() * 1.05
rate_lo = min(estimated_rates.values.min(), reconstructed_rates.values.min())
rate_hi = max(estimated_rates.values.max(), reconstructed_rates.values.max())

for i, t in enumerate(panel_times):
    emb_ax = bottom_axes[i // 2][(i % 2) * 2]
    rate_ax = bottom_axes[i // 2][(i % 2) * 2 + 1]

    # a) 2D embedding positions at this timepoint
    emb_t = nearest_embedding_time(t)
    slice_t = embeddings_df.loc[embeddings_df["time"] == emb_t].sort_values("individual_id")
    emb_ax.scatter(slice_t["X1"], slice_t["X2"],
                   c=slice_t["individual_id"].map(individual_colors),
                   s=1, alpha=1)
    emb_ax.set_xlim(-emb_lim, emb_lim)
    emb_ax.set_ylim(-emb_lim, emb_lim)
    emb_ax.set_aspect("equal")
    # three ticks per axis is all that fits legibly at this panel size
    emb_ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2, prune=None))
    emb_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2, prune=None))
    emb_ax.set_xlabel("$X_1$", fontsize=AXIS_LABEL_SIZE, labelpad=X_AXIS_LABEL_PAD)
    # labelpad bottoms out at the tick labels' width, so X2 is placed directly in axes
    # coordinates instead to sit tight against the spine
    emb_ax.set_ylabel("$X_2$", fontsize=AXIS_LABEL_SIZE)
    emb_ax.yaxis.set_label_coords(Y_LABEL_X, 0.5)

    # insets zooming into each cluster: the clusters are only a few hundred units wide
    # inside a +/-1470 view, so they are otherwise unreadable blobs. Once the groups have
    # merged into one cluster a single inset covers both.
    slice_t = slice_t.assign(group=slice_t["individual_id"].map(individual_groups))
    if t == panel_times[-1]:
        inset_groups = [(INSET_GROUP_KEYS, INSET_RECTS["merged"], INSET_CORNERS["merged"])]
    else:
        inset_groups = [([g], INSET_RECTS[g], INSET_CORNERS[g]) for g in INSET_GROUP_KEYS]

    for groups, rect, corners in inset_groups:
        members = slice_t.loc[slice_t["group"].isin(groups)]
        inset_ax = emb_ax.inset_axes(rect)
        inset_ax.scatter(members["X1"], members["X2"],
                         c=members["individual_id"].map(individual_colors),
                         s=5, alpha=1, linewidths=0)
        # limits hug the cluster, padded so no marker is clipped by the inset border
        x_lo, x_hi = members["X1"].min(), members["X1"].max()
        y_lo, y_hi = members["X2"].min(), members["X2"].max()
        pad = INSET_PAD_FRAC * max(x_hi - x_lo, y_hi - y_lo)
        inset_ax.set_xlim(x_lo - pad, x_hi + pad)
        inset_ax.set_ylim(y_lo - pad, y_hi + pad)
        inset_ax.set_aspect("equal")
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        # the house style hides the top/right spines, but an inset needs a closed box
        # to read as a zoomed frame
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
        # matplotlib picks the connecting corners itself, and with a diagonally opposite
        # inset it often joins edges that make the lines cross the panel; pin them instead
        _, connectors = emb_ax.indicate_inset_zoom(
            inset_ax, edgecolor="0.4", linewidth=0.4, alpha=0.8)
        for idx, connector in enumerate(connectors):
            connector.set_visible(idx in corners)

    # the timepoint titles the pair, so it is centred over both panels (in figure
    # coords) rather than over the left panel alone
    pair_left = emb_ax.get_position().x0
    pair_right = rate_ax.get_position().x1
    fig.text((pair_left + pair_right) / 2, emb_ax.get_position().y1 + 0.008,
             f"{t} generations ago", ha="center", va="bottom", fontsize=10)

    # b) estimated vs reconstructed rates for all pairs at this timepoint, colored by
    # pair type; between-group pairs dominate in number so they are drawn underneath
    col = nearest_rate_idx(t)
    for pair_type in ("between", "g0", "g1"):
        sel = pair_types == pair_type
        rate_ax.scatter(estimated_rates.iloc[:, col][sel], reconstructed_rates.iloc[:, col][sel],
                        s=4, color=PAIR_COLORS[pair_type], alpha=0.5, linewidths=0)
    rate_ax.plot([rate_lo, rate_hi], [rate_lo, rate_hi], color="black", lw=0.8, zorder=3)
    rate_ax.set_xscale("log")
    rate_ax.set_yscale("log")
    rate_ax.set_xlim(rate_lo, rate_hi)
    rate_ax.set_ylim(rate_lo, rate_hi)
    rate_ax.set_aspect("equal")
    # decade ticks only, labelled as powers of ten; the minor decade subdivisions
    # are too dense to read at this panel size
    for axis in (rate_ax.xaxis, rate_ax.yaxis):
        axis.set_major_locator(ticker.LogLocator(base=10.0))
        axis.set_major_formatter(ticker.LogFormatterSciNotation(base=10.0))
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())
    rate_ax.set_xlabel("Estimated rates", fontsize=AXIS_LABEL_SIZE, labelpad=X_AXIS_LABEL_PAD)
    rate_ax.set_ylabel("Reconstructed rates", fontsize=AXIS_LABEL_SIZE, labelpad=X_AXIS_LABEL_PAD)

    # the house style's 9pt labels and 4pt ticks are sized for full-page axes and
    # collide with the neighbouring panel at this size, so shrink them
    for ax in (emb_ax, rate_ax):
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE,
                       length=TICK_LENGTH, width=0.8, pad=1.5)

# key for the pair types, shown once on the first rate panel; drawn with opaque markers
# so the colours read clearly despite the scatter's alpha
pair_legend_handles = [
    Line2D([], [], marker="o", linestyle="none", markersize=2.5,
           color=PAIR_COLORS["g0"], label="Rates within Group A"),
    Line2D([], [], marker="o", linestyle="none", markersize=2.5,
           color=PAIR_COLORS["g1"], label="Rates within Group B"),
    Line2D([], [], marker="o", linestyle="none", markersize=2.5,
           color=PAIR_COLORS["between"], label="Rates between Groups"),
]
# centred on the gap between the two middle panels of the top row (nudged right of true
# centre), sitting just above them; a figure-level legend is used because the target
# spans two axes
LEGEND_X_SHIFT = 0.03
gap_centre_x = (bottom_axes[0][1].get_position().x1 + bottom_axes[0][2].get_position().x0) / 2 + LEGEND_X_SHIFT
fig.legend(
    handles=pair_legend_handles, loc="lower center",
    bbox_to_anchor=(gap_centre_x, bottom_axes[0][1].get_position().y1 + 0.005),
    frameon=False, fontsize=8, handlelength=1.0, handletextpad=0.3,
    labelspacing=0.25, borderpad=0.2,
)

# %% panel a: schematic of the demography
# migration rates taken from the demes file: the two groups differ 5-fold in their
# within-group rate, and the groups exchange migrants ~4 orders of magnitude more rarely
M_WITHIN_G0 = 0.001
M_WITHIN_G1 = 0.005
M_BETWEEN = 1e-07
ANCESTRAL_NE = 5000
GRID_N = 5   # each group is a GRID_N x GRID_N lattice of demes

MIG_COLORS = {"g0": "red", "g1": "blue", "between": "purple"}
GROUP_NAMES = {"g0": "Group A", "g1": "Group B"}

# Drawn in a 0-1 x 0-1 space with equal aspect, so deme circles stay circular. The axis
# is taller than it is wide, so the drawing is sized by WIDTH and the spare height is
# absorbed by the ancestral/split section above the grids.
schematic_ax.set_xlim(0, 1)
schematic_ax.set_ylim(0, 1)
schematic_ax.set_aspect("equal")
schematic_ax.axis("off")

# --- horizontal budget: two boxes plus the gap holding the between-group arrow ---
# EDGE_INSET keeps the outer box borders inside the axis limits; without it they sit
# exactly on x=0/x=1 and their stroke is clipped in half, so they look missing
EDGE_INSET = 0.012
BOX_GAP = 0.13        # space between the two boxes, wide enough for the arrow
box_w = (1.0 - 2 * EDGE_INSET - BOX_GAP) / 2   # each box spans this much width
grid_centers = {"g0": EDGE_INSET + box_w / 2, "g1": 1.0 - EDGE_INSET - box_w / 2}

# circle centres sit inside the box, inset by the padding that keeps circles off the edge
spacing = box_w / GRID_N                     # one lattice cell
deme_r = spacing * 0.34
grid_half = (box_w - spacing) / 2            # half-extent of the circle centres

# --- vertical layout, anchored at the TOP of the axis and stacked downwards, so the
# schematic starts flush with the top of the figure and any spare height falls below it
BRANCH_LEN = 0.10                            # length of the two vertical split arrows
STEM_LEN = 0.05                              # stem from the circle down to the connector
TOP_INSET = 0.012                            # keeps the circle's stroke inside the axis
anc_r = 0.085

box_half = grid_half + deme_r + 0.02         # half-height/width of the box itself
anc_xy = (0.5, 1.0 - TOP_INSET - anc_r)
split_y = anc_xy[1] - anc_r - STEM_LEN       # horizontal connector
box_top = split_y - BRANCH_LEN               # top edge of each box (arrow head lands here)
grid_mid_y = box_top - box_half              # centre of each grid/box
legend_y = box_top - 2 * box_half - 0.075    # under the boxes, clearing the group names

schematic_ax.add_patch(plt.Circle(anc_xy, anc_r, facecolor="0.75", edgecolor="black",
                                  lw=0.8, zorder=3))
schematic_ax.text(*anc_xy, f"$N_e$={ANCESTRAL_NE}", ha="center", va="center",
                  fontsize=8, zorder=4)

# ---- split: down from the ancestral deme, out to each side, then down into each group ----
# each descending branch is one continuous arrow from the connector to the box, so there
# is no seam between a plain line and a separate arrow below it
schematic_ax.plot([anc_xy[0], anc_xy[0]], [anc_xy[1] - anc_r, split_y], color="black", lw=0.9)
schematic_ax.plot([grid_centers["g0"], grid_centers["g1"]], [split_y, split_y], color="black", lw=0.9)
for x in grid_centers.values():
    schematic_ax.annotate("", xy=(x, box_top), xytext=(x, split_y),
                          arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9,
                                          shrinkA=0, shrinkB=0, mutation_scale=8))
SPLIT_TIME = 5000
schematic_ax.text(anc_xy[0], split_y - 0.012, f"{SPLIT_TIME} generations ago",
                  ha="center", va="top", fontsize=10)

# ---- the two 5x5 deme grids ----
for group, cx in grid_centers.items():
    xs = cx - grid_half + spacing * np.arange(GRID_N)
    ys = grid_mid_y + grid_half - spacing * np.arange(GRID_N)

    # within-group migration: nearest-neighbour lattice edges only, matching the demes file.
    # migration is symmetric, so each edge is a double-headed arrow; the ends stop at the
    # circle borders (offset by deme_r) to keep both heads visible
    # the visible shaft is only ~a third of a lattice cell (the rest is under the two
    # circles), so the heads are kept small and narrow or they swallow the whole edge
    edge_arrow = dict(arrowstyle="<|-|>,head_length=0.3,head_width=0.2",
                      color=MIG_COLORS[group], lw=0.7,
                      shrinkA=0, shrinkB=0, mutation_scale=2)
    for row in range(GRID_N):
        for col in range(GRID_N):
            if col < GRID_N - 1:
                schematic_ax.annotate("", xy=(xs[col + 1] - deme_r, ys[row]),
                                      xytext=(xs[col] + deme_r, ys[row]),
                                      arrowprops=edge_arrow, zorder=1)
            if row < GRID_N - 1:
                schematic_ax.annotate("", xy=(xs[col], ys[row + 1] + deme_r),
                                      xytext=(xs[col], ys[row] - deme_r),
                                      arrowprops=edge_arrow, zorder=1)

    # one circle per deme, colored to match the embeddings panels
    for row in range(GRID_N):
        for col in range(GRID_N):
            pop = f"{group}_{row}_{col}"
            schematic_ax.add_patch(plt.Circle(
                (xs[col], ys[row]), deme_r, facecolor=color_by_pop[pop],
                edgecolor="black", lw=0.4, zorder=2))

    # box around the group, named underneath; the migration rate it carries is in the legend
    schematic_ax.add_patch(plt.Rectangle(
        (cx - box_half, grid_mid_y - box_half), 2 * box_half, 2 * box_half,
        fill=False, edgecolor="black", lw=1.0, zorder=1))
    schematic_ax.text(cx, grid_mid_y - box_half - 0.015, GROUP_NAMES[group],
                      ha="center", va="top", fontsize=8)

# ---- between-group migration: double-headed arrow spanning the gap between the boxes ----
schematic_ax.annotate(
    "", xy=(grid_centers["g1"] - box_half, grid_mid_y),
    xytext=(grid_centers["g0"] + box_half, grid_mid_y),
    arrowprops=dict(arrowstyle="<|-|>", color=MIG_COLORS["between"],
                    lw=1.0, mutation_scale=7, shrinkA=0, shrinkB=0),
)

# ---- legend: one entry per arrow colour, giving the rate it represents ----
def sci_notation(rate):
    """Format a rate as m x 10^k for the legend."""
    exponent = int(np.floor(np.log10(rate)))
    mantissa = rate / 10 ** exponent
    return rf"${mantissa:g} \times 10^{{{exponent}}}$"

class DoubleArrowHandler(HandlerPatch):
    """Draw a legend key as a double-headed arrow, matching the migration arrows."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        arrow = FancyArrowPatch(
            (-xdescent, height / 2 - ydescent),
            (width - xdescent, height / 2 - ydescent),
            arrowstyle="<|-|>,head_length=2.5,head_width=1.5",
            color=orig_handle.get_edgecolor(), lw=1.0,
            shrinkA=0, shrinkB=0, mutation_scale=1,
        )
        arrow.set_clip_on(False)
        arrow.set_transform(trans)
        return [arrow]

# the handles only carry colour and label; DoubleArrowHandler does the drawing
legend_handles = [
    FancyArrowPatch((0, 0), (1, 0), color=MIG_COLORS["g0"],
                    label=f"m between neighbouring demes of group A = {sci_notation(M_WITHIN_G0)}"),
    FancyArrowPatch((0, 0), (1, 0), color=MIG_COLORS["g1"],
                    label=f"m between neighbouring demes of group B = {sci_notation(M_WITHIN_G1)}"),
    FancyArrowPatch((0, 0), (1, 0), color=MIG_COLORS["between"],
                    label=f"m between all demes of different groups = {sci_notation(M_BETWEEN)}"),
]
schematic_ax.legend(
    handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, legend_y),
    frameon=False, fontsize=8, ncol=1, handlelength=1.8, handletextpad=0.4,
    columnspacing=0.8, borderpad=0,
    handler_map={FancyArrowPatch: DoubleArrowHandler()},
)

# ---- subfigure labels, one per group of panels ----
# placed in figure coords from each group's own top-left corner
SUBFIG_LABEL_SIZE = 12

def add_subfig_label(letter, ax, dx=-0.005, dy=0.005, y=None):
    """
    Put a bold subfigure letter at the top-left of ax, in figure coordinates.
    y overrides the vertical position, to line a label up with a neighbouring one.
    """
    pos = ax.get_position()
    baseline = pos.y1 if y is None else y
    fig.text(pos.x0 + dx, baseline + dy, letter, ha="left", va="bottom",
             fontsize=SUBFIG_LABEL_SIZE, fontweight="bold")

add_subfig_label("A", schematic_ax)
add_subfig_label("C", bottom_axes[0][0], dy=0.022)

img, _ = plotter.save_static(**RENDER_KW)
img.save("../sim_figs_png/double_2d.png")
embeddings_ax.set_axis_off()
embeddings_ax.imshow(img, aspect="equal")
# B shares A's baseline so the two labels line up across the top row
add_subfig_label("B", embeddings_ax, y=schematic_ax.get_position().y1)
fig.savefig(f"../sim_figs/{filename}.pdf")
# %%
