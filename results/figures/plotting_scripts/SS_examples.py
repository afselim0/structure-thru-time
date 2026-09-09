#%%
"""
The 1D and 2D stepping-stone examples, one figure each (SS_1d.pdf, SS_2d.pdf).

Each figure is two vertical sections (p=1 left, p=2 right); each section is a 3x2
grid of (3D trajectories, 2D slice) with one migration rate per row.

Data: results/figure_data/SS_examples/, written by
utils/runs/prepare_figure_data/SS_examples.py -- the embeddings there are already
Procrustes-aligned within each p, so the rows differ by the demography rather than
by the optimizer's arbitrary rotation of the latent space.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
filename = os.path.basename(__file__).split(".")[0]
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
from PlotEmbeddings import *

FIGURE = "SS_examples"


def load_figure(name):
    """The aligned embeddings, labels and colours for one stepping-stone figure."""
    embeddings = pd.read_parquet(figure_data(FIGURE, f"{name}_embeddings.parquet"))
    labels_df = pd.read_csv(figure_data(FIGURE, f"{name}_pop_labels.csv"))
    colors_df = pd.read_csv(figure_data(FIGURE, f"{name}_pop_colors.csv"))
    # split the long table back into the per-(p, m) frames the panels are built from
    by_panel = {key: group.drop(columns=["p", "m"])
                for key, group in embeddings.groupby(["p", "m"], sort=False)}
    return by_panel, labels_df, colors_df

# %% figure-wide layout constants
# The only size parameter: the 2D slice panels are PANEL_SIZE x PANEL_SIZE squares, and
# every other dimension follows. Each 3D panel is PANEL_SIZE tall and PANEL_SIZE *
# aspect_ratio wide (aspect_ratio measured from the rendered cube), so a section is
# PANEL_SIZE * (1 + aspect_ratio) wide and the figure is
#   3 * PANEL_SIZE  tall  x  2 * PANEL_SIZE * (1 + aspect_ratio)  wide,
# plus the ROW_GAP/COL_GAP/SECTION_GAP separators and the margins carrying the labels.
PANEL_SIZE = 2.0

N = 100                                  # deme size, used to label rows by Nm
MIGRATION_RATES = [0.05, 0.01, 0.001]   # one row per rate, most to least migration
P_VALUES = [1, 2]                        # left section is p=1, right section is p=2

# the alignment reference used per figure is recorded in the prep script's params.csv;
# the embeddings read here are already aligned.

# Earliest time shown. The 3D panels' time axis starts here and the 2D panels show this
# exact slice, so the square is the bottom face of the cube beside it. Set well clear of
# t=0, where the rate estimator has almost no coalescent events to work from and the
# embedding is correspondingly unconstrained.
MIN_TIME = 50.0

# which timepoint the 2D slice panels show; None falls back to the smallest non-zero time
SLICE_TIME = MIN_TIME

# margins reserved around the 3x2 panel block, in inches. The left margin carries the
# rotated Nm row labels and the top carries the "p = 1"/"p = 2" section titles.
LEFT_MARGIN = 0.28
# the 2D panels write their shared x-axis exponent just past their right edge, so the
# right margin has to clear it as well as the p=2 section's tick labels
RIGHT_MARGIN = 0.50
TOP_MARGIN = 0.26
BOTTOM_MARGIN = 0.34
# gaps between panels, all in inches so they do not scale with PANEL_SIZE. COL_GAP is
# sized for the 2D panels' y tick labels and SECTION_GAP for the p=1 panel's x-axis
# exponent; both are drawn outside the axes box and would otherwise land on the next cube.
SECTION_GAP = 0.46     # horizontal space between the p=1 and p=2 sections
COL_GAP = 0.30         # horizontal space between the 3D and the 2D-slice column
ROW_GAP = 0.25         # vertical space between migration-rate rows

# text sizes for PANEL_SIZE ~2in panels: below the house style's 9-11pt, which is set for
# full-page axes, but larger than the 5-6pt these panels used when they were ~1.3in wide
TICK_LABEL_SIZE = 7
AXIS_LABEL_SIZE = 8

# --- 3D panel axis text: RENDER_PX is the knob ---
# Plotly lays fonts out in fixed pixels, so what sets the text-to-cube proportion is the
# size of the export canvas, not any font size. The cube is rendered on a RENDER_PX square
# canvas and then scaled back into the panel, so the text ends up smaller relative to the
# cube the larger this is:
#   RENDER_PX up   -> smaller 3D text (bigger canvas, same fonts)
#   RENDER_PX down -> larger 3D text
# It does not change the printed panel size, only the text/plot ratio. Note this also
# thins the trajectory lines proportionally -- compensate with line_width in the
# add_sample_trajectories call below. Cost is quadratic in render time.
# 1000 is PANEL_SIZE=2.0 at 500 dpi, i.e. Plotly's untouched proportion for this panel.
# The cropped aspect ratio depends on this value, so the probe render that sizes the
# panels below uses it too.
RENDER_PX = 1000
N_XY_TICKS = 5
N_TIME_TICKS = 5
N_SLICE_TICKS = 3      # ticks per axis on the 2D slice panels
ROW_LABEL_SIZE = 10
SECTION_TITLE_SIZE = 12
SUBFIG_LABEL_SIZE = 14   # the bold A/B naming each half
TICK_LENGTH = 2.5
SLICE_MARKER_SIZE = 5.0


def sci_notation(value):
    """
    Format a number as scientific notation for a Plotly 3D tick label.

    Plotly's 3D scene tick labels render neither LaTeX nor HTML tags, so the exponent is
    written with Unicode superscript digits. Exact powers of ten come out as "10^k" with
    no redundant "1x" prefix; everything else keeps one decimal in the mantissa. Zero has
    no exponent and is returned as "0".
    """
    if value == 0:
        return "0"
    superscripts = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0 ** exponent
    power = f"10{str(exponent).translate(superscripts)}"
    # treat as an exact power of ten if the mantissa rounds to +/-1 at the shown precision
    if abs(abs(mantissa) - 1) < 0.05:
        return f"-{power}" if mantissa < 0 else power
    return f"{mantissa:.1f}×{power}"


def slice_time(embeddings_df):
    """
    Timepoint the 2D slice panels show: SLICE_TIME, or the smallest non-zero time.

    SLICE_TIME is snapped to the stored time grid, which is not the same across figures
    (SS_1d steps by 20 and SS_2d by 10), so a requested time need not exist in both. The
    snap is reported when it moves the slice, since the panel would otherwise be labelled
    with a time it does not show.
    """
    times = np.sort(embeddings_df["time"].unique())
    if SLICE_TIME is None:
        nonzero = times[times > 0]
        if nonzero.size == 0:
            raise ValueError("Embeddings contain no non-zero timepoint for the 2D slice")
        return float(nonzero[0])
    nearest = float(times[np.argmin(np.abs(times - SLICE_TIME))])
    if not np.isclose(nearest, SLICE_TIME):
        print(f"  note: t={SLICE_TIME:g} is not on this time grid; using t={nearest:g}")
    return nearest


def individual_color_map(labels_df, colors_df):
    """Map each individual_id to its population's color, for the 2D slice scatter."""
    color_by_pop = dict(zip(colors_df["population"], colors_df["color"]))
    return (
        labels_df.drop_duplicates("individual_id")
        .set_index("individual_id")["population"]
        .map(color_by_pop)
    )


def make_SS_figure(embeddings, labels_df, colors_df, out_name):
    """
    Build one SS figure: two vertical sections (p=1 left, p=2 right), each a 3x2 grid
    of (3D trajectories, 2D slice) with one migration rate per row.

    embeddings maps (p, m) -> embeddings DataFrame.
    """
    individual_colors = individual_color_map(labels_df, colors_df)

    # the embeddings arrive Procrustes-aligned within each p (see the prep script), so
    # the rows are already comparable and are used as they come
    aligned = embeddings

    # the 3D panels' width follows their true rendered aspect ratio, so build every
    # plotter up front: the panel geometry below is derived from the aspect ratio, and
    # the render needs the finished Plotly figure anyway
    plotters = {}
    for p in P_VALUES:
        for m in MIGRATION_RATES:
            plotter = PlotEmbeddings(
                embeddings_df=aligned[(p, m)],
                labels_df=labels_df,
                colors_df=colors_df,
                sample_id_col="individual_id",
            )
            # each panel gets its own limit from its own data, rather than one shared
            # across the figure. It is passed explicitly (instead of letting init_plot
            # derive a padded data min/max) so the range stays symmetric about zero: the
            # ticks are then centred on the origin and the drift of the embedding away
            # from it stays readable. The matching 2D slice reuses this range below.
            # measured over the visible times only: the pre-MIN_TIME positions are the
            # most spread out (the estimator is least constrained there) and are no longer
            # drawn, so including them would zoom every panel out by up to ~2x. The floor
            # is the slice's own time, not MIN_TIME: SS_1d's grid has no t=50, so its
            # slice snaps to t=40 and would otherwise be clipped by the tighter limit.
            visible_from = min(MIN_TIME, slice_time(aligned[(p, m)]))
            visible = aligned[(p, m)].loc[aligned[(p, m)]["time"] >= visible_from]
            panel_lim = np.abs(visible[["X1", "X2"]].to_numpy()).max() * 1.05
            plotter.init_plot(xy_lims=panel_lim, min_time=MIN_TIME)
            plotter.add_sample_trajectories(line_width=1.0, alpha=0.5)
            # X/Y ticks are pinned explicitly: nticks is only an upper hint, and Plotly
            # then picks its own "nice" 1/2/5 x 10^n interval, so the count it settles on
            # follows the data range -- the same nticks gave 5 ticks in one figure and 7
            # in the other. Font size is left at its default: the text/plot proportion is
            # set by RENDER_PX at export instead (see the panel renders below).
            # read back the range init_plot derived for this panel, so the ticks span
            # exactly the cube that is drawn
            xy_lo, xy_hi = plotter._plot_ranges.x_range
            xy_ticks = np.linspace(xy_lo, xy_hi, N_XY_TICKS + 2)[1:-1]
            xy_axis = dict(
                tickvals=xy_ticks,
                ticktext=[sci_notation(v) for v in xy_ticks],
            )
            # Time: whole decades only, so the labels read 10^2, 10^3, ... rather than the
            # 3.16e2-style values that come of splitting the range into equal log steps.
            # The axis starts at MIN_TIME, so only decades at or above it are labelled.
            z0, z1 = plotter._plot_ranges.z_range
            z_ticks = 10.0 ** np.arange(
                np.ceil(np.log10(z0)), np.floor(np.log10(z1)) + 1)
            plotter.fig.update_layout(
                scene=dict(
                    xaxis=dict(title="X1", **xy_axis),
                    yaxis=dict(title="X2", **xy_axis),
                    zaxis=dict(
                        title="Time",
                        tickvals=z_ticks,
                        ticktext=[sci_notation(t) for t in z_ticks],
                    ),
                ),
            )
            plotters[(p, m)] = plotter

    # ---- panel geometry, in inches ----
    # The figure size is derived from PANEL_SIZE rather than fixed up front: the cube's
    # true aspect ratio decides how wide a 3D panel has to be, and forcing that into a
    # preset figure width is what left dead space between the panels. Every panel is
    # PANEL_SIZE tall; the 2D slice is a PANEL_SIZE square and the cube is
    # PANEL_SIZE * aspect_ratio wide, so each abuts its neighbour exactly.
    # All six cubes share one ratio (they differ only in data extent), so one probe render
    # sizes them all.
    aspect_ratio = plotters[(P_VALUES[0], MIGRATION_RATES[0])].save_static(
        width=RENDER_PX, height=RENDER_PX, scale=1)[1]  # scale does not affect the ratio

    panel_2d_w = panel_2d_h = panel_3d_h = PANEL_SIZE
    panel_3d_w = PANEL_SIZE * aspect_ratio
    row_h = PANEL_SIZE
    row_gap = ROW_GAP                      # fixed inches, so it does not scale with PANEL_SIZE
    col_gap = COL_GAP                      # fixed inches, like row_gap
    top_offset = 0.0

    # a section holds one cube plus one square; the grid is 3 rows by 2 sections
    section_w = panel_3d_w + col_gap + panel_2d_w
    grid_w = 2 * section_w + SECTION_GAP
    grid_h = 3 * row_h + 2 * row_gap

    # the margins carry the row labels, section titles and the 2D panels' tick labels;
    # they sit outside the panel block, so the panels themselves stay flush
    FIG_W = grid_w + LEFT_MARGIN + RIGHT_MARGIN
    FIG_H = grid_h + TOP_MARGIN + BOTTOM_MARGIN

    # constrained_layout is on in the house style, but every axis below is pinned to an
    # explicit inch box; leaving it on makes matplotlib try to relayout them and warn
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=500)
    fig.set_layout_engine("none")

    # axes placed at explicit inch coordinates: a GridSpec would divide its box evenly
    # and could not give the 3D and 2D columns the different widths derived above
    axes = {}
    for section, p in enumerate(P_VALUES):
        # the sections tile the grid exactly: each is one cube plus one square wide
        section_x = LEFT_MARGIN + section * (section_w + SECTION_GAP)
        for row, m in enumerate(MIGRATION_RATES):
            # rows run top to bottom; y is measured from the bottom of the figure
            row_y = (FIG_H - TOP_MARGIN - top_offset
                     - (row + 1) * row_h - row * row_gap)
            # the cube starts at the section's left edge and the 2D square sits flush
            # against its right edge, so the two panels share an edge with no gap
            panel_2d_x = section_x + panel_3d_w + col_gap
            # zorder puts the 2D panels above the cubes: the cube renders are placed last
            # and can spill slightly past their box, and the slice must stay legible
            # where they overlap
            axes[(p, m, "3d")] = fig.add_axes([
                section_x / FIG_W, row_y / FIG_H,
                panel_3d_w / FIG_W, panel_3d_h / FIG_H,
            ], zorder=1)
            axes[(p, m, "2d")] = fig.add_axes([
                panel_2d_x / FIG_W, row_y / FIG_H,
                panel_2d_w / FIG_W, panel_2d_h / FIG_H,
            ], zorder=2)

    # ---- 2D slice panels: positions at a single timepoint ----
    # each panel scales to its own data: the spatial extent of the embedding changes by
    # orders of magnitude across migration rates, so a shared range would collapse the
    # low-migration slices to a dot
    for p in P_VALUES:
        for m in MIGRATION_RATES:
            ax = axes[(p, m, "2d")]
            df = aligned[(p, m)]
            t = slice_time(df)
            slice_df = df.loc[df["time"] == t].sort_values("individual_id")
            ax.scatter(
                slice_df["X1"], slice_df["X2"],
                c=slice_df["individual_id"].map(individual_colors),
                s=SLICE_MARKER_SIZE, linewidths=0,
            )
            # the slice takes the same range as the 3D panel beside it, so the pair reads
            # on one scale and a point sits at the same place in both. init_plot gives X1
            # and X2 a common range, so the square panel keeps an equal aspect.
            xy_lo, xy_hi = plotters[(p, m)]._plot_ranges.x_range
            ax.set_xlim(xy_lo, xy_hi)
            ax.set_ylim(xy_lo, xy_hi)
            ax.set_aspect("equal", adjustable="box")
            # Ticks are pinned explicitly rather than left to a locator: MaxNLocator only
            # keeps its "nice" round values that fall inside the limits, so the outermost
            # one could land outside and the panel silently showed 2 ticks instead of 3.
            # Inset from the axis ends so the labels do not collide with the next panel.
            ticks = np.linspace(xy_lo, xy_hi, N_SLICE_TICKS + 2)[1:-1]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            # the shared 1eN exponent is folded out of the tick labels so the (large)
            # coordinate values stay short
            # the exponent comes from the axis range rather than the slice's own values:
            # the range spans every timepoint, so it is the larger of the two, and taking
            # it from the slice would scale the labels against a number the panel does
            # not show
            exponent = int(np.floor(np.log10(max(abs(xy_lo), abs(xy_hi)))))
            # rounding before formatting turns a centre tick that sits a hair below zero
            # into a true 0.0, so it prints as "0.0" rather than "-0.0"
            scaled = ticker.FuncFormatter(
                lambda v, _pos, e=exponent: f"{round(v / 10 ** e, 1) + 0.0:.1f}")
            ax.xaxis.set_major_formatter(scaled)
            ax.yaxis.set_major_formatter(scaled)
            # both axes are divided by the same exponent, but it is written on each of
            # them -- one label above the y axis is easy to read as applying only to y
            ax.text(0.0, 1.02, f"$\\times 10^{{{exponent}}}$", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=TICK_LABEL_SIZE)
            ax.text(1.02, 0.0, f"$\\times 10^{{{exponent}}}$", transform=ax.transAxes,
                    ha="left", va="center", fontsize=TICK_LABEL_SIZE)
            ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE,
                           length=TICK_LENGTH, width=0.6, pad=1.0)
            # an opaque face, so a cube render that spills past its own box is hidden
            # behind this panel rather than showing through it
            ax.set_facecolor("white")
            ax.patch.set_alpha(1.0)
            # COL_GAP opens enough room to the panel's left for the y tick labels, so the
            # axis stays on its natural side and every panel keeps its labels
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            # ROW_GAP leaves room under every row, so all rows keep their x tick labels;
            # only the bottom row is named, since the columns all show the same quantity
            if m == MIGRATION_RATES[-1]:
                ax.set_xlabel("$X_1$", fontsize=AXIS_LABEL_SIZE, labelpad=1.0)
            
            ax.set_ylabel("$X_2$", fontsize=AXIS_LABEL_SIZE, labelpad=1.0)

    # ---- row labels (migration rate) and section titles (p) ----
    # placed in figure coords off each block's own edge
    # rows are labelled by Nm, the scaled migration rate that actually governs how much
    # structure the demography retains, rather than by m alone
    # the label sits just left of the first 3D panel rather than at the figure edge, so it
    # stays close to the row it names however wide the left margin is
    ROW_LABEL_PAD = 0.01   # figure-fraction gap between the label and the panel
    for m in MIGRATION_RATES:
        pos = axes[(P_VALUES[0], m, "3d")].get_position()
        fig.text(pos.x0 - ROW_LABEL_PAD, (pos.y0 + pos.y1) / 2, f"$Nm = {N * m:g}$",
                 rotation=90, ha="right", va="center", fontsize=ROW_LABEL_SIZE)

    # a bold subfigure letter per section, at its top-left corner: p=1 is A, p=2 is B
    for section, p in enumerate(P_VALUES):
        pos = axes[(p, MIGRATION_RATES[0], "3d")].get_position()
        fig.text(pos.x0, pos.y1 + 0.012, chr(ord("A") + section), ha="left", va="bottom",
                 fontsize=SUBFIG_LABEL_SIZE, fontweight="bold")

    for p in P_VALUES:
        left = axes[(p, MIGRATION_RATES[0], "3d")].get_position().x0
        right = axes[(p, MIGRATION_RATES[0], "2d")].get_position().x1
        top = axes[(p, MIGRATION_RATES[0], "3d")].get_position().y1
        fig.text((left + right) / 2, top + 0.012, f"$p = {p}$",
                 ha="center", va="bottom", fontsize=SECTION_TITLE_SIZE)

    # ---- 3D panels last, so they sit above the 2D slices in draw order ----
    for p in P_VALUES:
        for m in MIGRATION_RATES:
            # each box is exactly PANEL_SIZE * aspect_ratio wide, and the render shares
            # RENDER_PX with the probe that produced that ratio, so it fills the box edge
            # to edge with no letterboxing
            img, _ = plotters[(p, m)].save_static(width=RENDER_PX, height=RENDER_PX, scale=1)
            img.save(f"../sim_figs_png/{out_name}_p{p}_m{str(m).split('.')[1]}.png")
            ax_3d = axes[(p, m, "3d")]
            ax_3d.set_axis_off()
            ax_3d.imshow(img, aspect="equal")

    # the house style sets savefig.bbox="tight", which crops the figure to its contents
    # and would change the total size away from the required FIG_W x FIG_H. The rcParam
    # has to be overridden here, since savefig(bbox_inches=None) just falls back to it.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(f"../sim_figs/{out_name}.pdf")
    return fig


# %% 1D stepping-stone
SS_1d_embeddings, SS_1d_pop_labels, SS_1d_pop_colors = load_figure("SS_1d")
SS_1d_fig = make_SS_figure(
    embeddings=SS_1d_embeddings,
    labels_df=SS_1d_pop_labels,
    colors_df=SS_1d_pop_colors,
    out_name="SS_1d",
)

# %% 2D stepping-stone
SS_2d_embeddings, SS_2d_pop_labels, SS_2d_pop_colors = load_figure("SS_2d")
SS_2d_fig = make_SS_figure(
    embeddings=SS_2d_embeddings,
    labels_df=SS_2d_pop_labels,
    colors_df=SS_2d_pop_colors,
    out_name="SS_2d",
)
# %%
