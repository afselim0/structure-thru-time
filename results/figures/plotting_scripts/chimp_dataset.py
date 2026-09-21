#%%
"""
The chimpanzee dataset and what CoaSTT measures on it.

Panels
  A  origins of the sampled individuals on a map of equatorial Africa
                              <- africa_outlines.parquet (country, polygon_id, order,
                                 longitude, latitude), sample_locations.csv
  B  genotype PCA over the run's tree sequences
                              <- pca.csv (sample_id, PC1, PC2, PC*_explained)
  C  Ne(t) per subspecies: 1/(2*mean lambda) over that subspecies' within-pairs,
     over one thin line per within-pair
                              <- effective_size_mean.parquet, effective_size_pairs.parquet
  D  relative diversity rho_AB(t) = between / pooled within, for the six pairs among
     the four subspecies, over one thin line per between-pair
                              <- relative_divergence_mean.parquet,
                                 relative_divergence_pairs.parquet, pair_colors.csv

Data: results/figure_data/chimp_dataset/, written by
utils/runs/prepare_figure_data/chimp_dataset.py. Nothing is computed here: the
aggregation, the thinning of the individual trajectories and the axis limits all
happen there.
"""
import os

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from results.figures.plotting_style import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))

FIGURE = "chimp_dataset"
KIND = "app"

# ---------------------------------------------------------------------------
# Plotting parameters: the only knobs. Inches unless stated.
# ---------------------------------------------------------------------------
# The map spans the whole first row at equal aspect, so the figure's width sets the
# map's height: at the full 6.5 in the map would be 3.4 in tall and swamp the rest.
# 5.5 in keeps it to ~45% of the figure height and still leaves the two bottom
# panels square at 1.92 in. Narrower than the 6.5 in double-column maximum, which
# is allowed -- the limit is an upper bound.
WIDTH = 5.5
# Panels C and D carry two-line y labels, one of them a stacked fraction, so the
# strip has to be wider than a single line of text needs.
MARGIN_L = 0.78
MARGIN_R = 0.08
MARGIN_T = 0.12
MARGIN_B = 0.40         # x-axis label + tick labels
# Between the two bottom panels. Most of the visible gap is the right panel's own
# y label and tick labels, which need ~0.47 in whatever this is set to; 0.30 is
# about as tight as it goes before they crowd the left panel.
COL_GAP = 0.14
# The bottom row now carries a years axis and its label on top of each panel, so
# the gap between the rows has to hold that as well as the map's x label.
ROW_GAP = 0.62
# Bottom strip. It now carries panel D's pair key (the species colours moved into
# the map, where they are read against the ranges they describe).
# Six entries carrying "t_0.5 = x kya" do not fit on one 5.5 in row, so the key
# is two rows of three.
LEGEND_H = 0.52
LEGEND_NCOL = 2
# The species key sits in the map's top-right, which is empty of both samples and
# range polygons, in a white box so it reads over the land fill.
MAP_LEGEND_LOC = "upper right"

# The PCA is an inset in the map's Atlantic, given in map coordinates (degrees) so
# it moves with the map rather than with the figure. The box was chosen as the
# largest sample-free rectangle inside the map bounds: the whole south-west corner
# is ocean, and the nearest sample (Central, 10.0 E) lies outside its eastern edge.
PCA_INSET_BOX = dict(lon_min=-17.0, lon_max=6.0, lat_min=-12.0, lat_max=2.5)
# Fractions of the inset box kept for its own labels, which sit inside the box so
# the whole inset stays over open water.
PCA_INSET_LABEL_PAD = 0.26   # below the axes, for the x label and ticks
PCA_INSET_LEFT_PAD = 0.26    # left of the axes, for the y label and ticks

# Panel A. The country borders are gone: political boundaries say nothing about
# where chimpanzee lineages separate, so the map carries only the coast, the
# barriers that do separate the subspecies, and the ranges themselves.
# Basemap styling follows Steux et al. 2025 (Hum Popul Genet Genom 5(2):0004,
# Figure 1), which draws the same IUCN ranges over a Natural Earth background:
# cream land against a white ocean, so the saturated range fills carry the panel.
LAND_COLOR = "#F2EFE6"
OCEAN_COLOR = "#EAF2F8"     # a faint blue, so sea reads as sea against the land
COAST_COLOR = "#B9B3A6"
COAST_WIDTH = 0.5
# The four barriers. Rivers are drawn as real courses from Natural Earth; the
# Dahomey Gap is a savanna belt, so it is a band rather than a line. The two
# firmly supported subspecies boundaries (Sanaga, Ubangi) are drawn solid and the
# more equivocal ones (Niger, contested; Congo, a species-level boundary) dashed,
# so the map does not assert more than the evidence does.
RIVER_COLOR = "#1F3F8F"      # the deep navy Steux et al. use for the rivers
# Every river solid and the same weight: a dashed line read as a different kind of
# feature rather than as weaker evidence, which is not what it was meant to say.
# Which boundaries are firm and which are contested is a matter for the caption.
RIVER_WIDTH = 1.6
GAP_COLOR = "#C8A24B"
GAP_EDGE_COLOR = "#A8842F"
GAP_ALPHA = 0.45
# The ranges are context for the samples, not the subject, so they are tinted
# rather than filled: at higher opacity they overwhelmed the sample markers and
# the rivers drawn over them.
RANGE_ALPHA = 0.35
BARRIER_LABEL_SIZE = "base"
# Where each barrier is named, in map degrees. Placed off the feature itself into
# open space -- the Sanaga and Ubangi labels sit north of their courses, the Congo
# label south-east of the bend, and the Dahomey Gap above its band -- so no label
# sits on a sample point or on the feature it names.
BARRIER_LABELS = {
    # Each river is named on two lines, so the word sits beside its course rather
    # than running along it. The Gap's label sits just above its polygon, which
    # spans lon -2.0..3.5 and tops out at 8.9 N; the Niger's sits above the crown
    # of its arc, which peaks at 17.1 N.
    "Dahomey Gap": (-0.2, 10.4, "center"),
    "Sanaga\nRiver": (13.8, 5.6, "left"),
    "Ubangi\nRiver": (19.5, 4.5, "left"),
    "Congo\nRiver": (29.0, -10.0, "center"),
    "Niger\nRiver": (-8.2, 15.8, "center"),
}
# Marker area in points^2 for a single individual; points holding several
# individuals grow with the count, so area stays proportional to how many animals
# share an origin (1 to 6 here). The edge is kept, unlike the PCA inset's markers:
# map points sit as close as 0.4 deg and overlap at the Cameroon and DRC clusters,
# and the outline is what separates them from each other and from the land.
MARKER_AREA = 24.0
MARKER_EDGE = 0.4

# Panels C and D. One LineCollection per colour rather than one Line2D per
# trajectory: 1378 Line2D artists would be 1378 separate PDF path objects.
TRACE_STYLE = dict(linewidths=0.2, alpha=0.10, zorder=1)
MEAN_STYLE = dict(lw=1.4, zorder=4)
# The PCA points cluster very tightly within each subspecies -- the median
# nearest-neighbour distance is 0.04 deg in the aligned frame, and 96% of points
# sit closer to a neighbour than one marker width -- so they are drawn large,
# borderless and semi-transparent: overlap then reads as depth of colour, which is
# the honest way to show a cluster of near-identical individuals.
# Panels C and D start here rather than at the config's min_time of 50: below
# ~200 generations almost no pair has coalesced, so the curves there are dominated
# by the estimator's floor rather than by demography.
TIME_MIN_GENERATIONS = 200.0
# The reference level panel D marks, matching RHO_CROSSING_LEVEL in the prep script.
RHO_CROSSING_LEVEL = 0.5
# The years axis on top of C and D. de Manuel et al. 2016 assume 25 years per
# chimpanzee generation, which is also what params.csv records.
GENERATION_TIME = 25

PCA_MARKER = 34
PCA_MARKER_ALPHA = 0.55

# Panel D's colours encode a population PAIR, a different encoding from the shared
# per-subspecies legend, so the pairs are keyed inside the panel. The curves all
# rise to rho = 1 by the deep end, leaving the upper left empty for it.
PAIR_SHORT = {"Western": "W", "Eastern": "E", "Central": "C", "NC": "NC"}
# Panel D's key, in two columns of three: the pairs not involving NC on the left
# (Eastern-Central and the two Western pairs), the three NC pairs on the right.
# Within each column the entries run in order of crossing time.
PAIR_LEGEND_COLUMNS = [
    ["Eastern-Central", "Western-Eastern", "Western-Central"],
    ["Eastern-NC", "Western-NC", "Central-NC"],
]

HYBRID_POPULATION = "W x C"
POPULATION_ORDER = ["Western", "NC", "Central", "Eastern"]
LEGEND_ORDER = POPULATION_ORDER + [HYBRID_POPULATION]
# Bare labels: the subspecies binomials (verus, ellioti, troglodytes,
# schweinfurthii) do not fit on one 6.5 in line at 8 pt and belong in the caption.
BONOBO_POPULATION = "Bonobo"
POPULATION_LABELS = {"Western": "Western", "NC": "Nigeria-Cameroon",
                     "Central": "Central", "Eastern": "Eastern",
                     HYBRID_POPULATION: "W x C hybrid",
                     BONOBO_POPULATION: "Bonobo (range only)"}


# ---------------------------------------------------------------------------
# Data: read, index, nothing else.
# ---------------------------------------------------------------------------
def load_data():
    d = {}
    d["outlines"] = pd.read_parquet(figure_data(FIGURE, "africa_outlines.parquet"))
    d["rivers"] = pd.read_parquet(figure_data(FIGURE, "boundary_rivers.parquet"))
    d["dahomey"] = pd.read_parquet(figure_data(FIGURE, "dahomey_gap.parquet"))
    # The IUCN ranges are optional: they are not redistributable, so the prep
    # script only writes them when CHIMP_IUCN_RANGES pointed at a shapefile.
    ranges_path = figure_data(FIGURE, "subspecies_ranges.parquet")
    d["ranges"] = pd.read_parquet(ranges_path) if os.path.exists(ranges_path) else None
    d["locations"] = pd.read_csv(figure_data(FIGURE, "sample_locations.csv"))
    d["pop_colors"] = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
    d["pop_labels"] = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
    d["pca"] = pd.read_csv(figure_data(FIGURE, "pca.csv"))
    d["pca_aligned"] = pd.read_csv(figure_data(FIGURE, "pca_aligned.csv"))
    d["ne_mean"] = pd.read_parquet(figure_data(FIGURE, "effective_size_mean.parquet"))
    d["ne_pairs"] = pd.read_parquet(figure_data(FIGURE, "effective_size_pairs.parquet"))
    d["rho_mean"] = pd.read_parquet(figure_data(FIGURE, "relative_divergence_mean.parquet"))
    d["rho_pairs"] = pd.read_parquet(figure_data(FIGURE, "relative_divergence_pairs.parquet"))
    d["pair_colors"] = pd.read_csv(figure_data(FIGURE, "pair_colors.csv"))
    d["crossings"] = pd.read_csv(figure_data(FIGURE, "rho_crossings.csv"))

    params = pd.read_csv(figure_data(FIGURE, "params.csv"))
    d["params"] = dict(zip(params["parameter"], pd.to_numeric(params["value"])))
    d["color_by_pair"] = dict(zip(d["pair_colors"]["pair_label"], d["pair_colors"]["color"]))
    # Bonobo has no sample, so its colour travels in its own one-row table rather
    # than in pop_colors.csv, which is keyed to the sequenced populations.
    bonobo = pd.read_csv(figure_data(FIGURE, "bonobo_color.csv"))
    d["color_by_pop"] = {**dict(zip(d["pop_colors"]["population"], d["pop_colors"]["color"])),
                         **dict(zip(bonobo["population"], bonobo["color"]))}
    return d


def time_axes(ax, params):
    """
    Log time axis on the bottom in generations, a twin in years on top.

    log_axis sets decade majors and removes the minors; the minors are put back
    here because over three decades the unlabelled 2..9 ticks are what let a
    reader place a feature between the powers of ten. The twin shares the same
    limits scaled by the generation time, so the two axes cannot drift apart.
    """
    log_axis(ax, "x")
    log_axis(ax, "y")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                                 numticks=100))
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(which="minor", length=1.5, width=0.5)

    low, high = TIME_MIN_GENERATIONS, params["max_time"]
    ax.set_xlim(low, high)

    years = ax.secondary_xaxis(
        "top", functions=(lambda g: g * GENERATION_TIME,
                          lambda y: y / GENERATION_TIME))
    years.set_xscale("log")
    years.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
    years.xaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10.0))
    years.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=tuple(range(2, 10)),
                                                    numticks=100))
    years.xaxis.set_minor_formatter(ticker.NullFormatter())
    years.tick_params(labelsize=FONT_SIZES["tick"], which="minor",
                      length=1.5, width=0.5)
    years.set_xlabel(f"Time (years ago, g = {GENERATION_TIME:g})",
                     fontsize=FONT_SIZES["base"])
    return years


def _trace_times(frame, key_columns):
    """The time axis of a trajectory table, from its non-key column names."""
    return np.array([float(c) for c in frame.columns if c not in key_columns])


def add_trace_bundle(ax, times, values, color):
    """
    Draw one bundle of individual trajectories as a single LineCollection.

    LineCollection has no NaN handling of its own, so each trace is split into its
    runs of plottable points first -- non-finite (a floored pair, or a ratio with a
    NaN denominator) and non-positive values both have to break the line rather
    than be drawn, because both axes are log.
    """
    segments = []
    for row in values:
        usable = np.isfinite(row) & (row > 0)
        if not usable.any():
            continue
        # Boundaries of each contiguous run of usable points.
        edges = np.flatnonzero(np.diff(usable.astype(np.int8)))
        starts = np.concatenate([[0], edges + 1])
        stops = np.concatenate([edges + 1, [len(row)]])
        for start, stop in zip(starts, stops):
            if usable[start] and stop - start > 1:
                segments.append(np.column_stack([times[start:stop], row[start:stop]]))
    ax.add_collection(LineCollection(segments, colors=color, **TRACE_STYLE))


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_a(ax, d):
    """
    Sample origins on a map of equatorial Africa, over the barriers that separate
    the four subspecies and (when available) their ranges.

    Country borders are deliberately absent: they are political, and nothing about
    them predicts where chimpanzee lineages divide. What is drawn instead is the
    four features that do -- the Dahomey Gap, the widest and oldest barrier and the
    reason western chimpanzees are the most distinct lineage; the Sanaga, the best
    supported of the river boundaries; the Ubangi; and the Congo, which bounds the
    range to the south and separates chimpanzees from bonobos rather than one
    subspecies from another.
    """
    ax.set_facecolor(OCEAN_COLOR)
    # Land first, as a single flat colour: only the coast is an outline now.
    for _, ring in d["outlines"].groupby("polygon_id"):
        ax.add_patch(Polygon(ring[["longitude", "latitude"]].values,
                             closed=True, facecolor=LAND_COLOR,
                             edgecolor=COAST_COLOR, linewidth=COAST_WIDTH,
                             zorder=1))

    # Approximate subspecies ranges, tinted in each subspecies' own colour.
    if d["ranges"] is not None:
        for population, polygons in d["ranges"].groupby("population"):
            for _, ring in polygons.groupby("polygon_id"):
                ax.add_patch(Polygon(ring[["longitude", "latitude"]].values,
                                     closed=True,
                                     facecolor=d["color_by_pop"][population],
                                     edgecolor="none", alpha=RANGE_ALPHA,
                                     zorder=2))

    # The Dahomey Gap is a belt of savanna reaching the coast, not a line and not
    # a box: this is the real ecoregion polygon, so it follows the coastline and
    # stays on land.
    ax.add_patch(Polygon(d["dahomey"][["longitude", "latitude"]].values,
                         closed=True, facecolor=GAP_COLOR,
                         edgecolor=GAP_EDGE_COLOR, linewidth=0.4,
                         alpha=GAP_ALPHA, zorder=3))

    for _, segment in d["rivers"].groupby("segment_id"):
        ax.plot(segment["longitude"], segment["latitude"],
                color=RIVER_COLOR, lw=RIVER_WIDTH, solid_capstyle="round",
                zorder=4)

    locations = d["locations"]
    for population in POPULATION_ORDER:
        points = locations[locations["population"] == population]
        if points.empty:
            continue
        ax.scatter(points["longitude"], points["latitude"],
                   s=MARKER_AREA * points["n_individuals"],
                   facecolor=d["color_by_pop"][population],
                   edgecolor="black", linewidth=MARKER_EDGE,
                   zorder=6, clip_on=True)

    ax.set_xlim(d["params"]["map_lon_min"], d["params"]["map_lon_max"])
    ax.set_ylim(d["params"]["map_lat_min"], d["params"]["map_lat_max"])
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.tick_params(labelsize=FONT_SIZES["tick"])

    # The species key lives here rather than under the figure: colour means the
    # same thing in A, B and C, and in the map it can be read directly against the
    # ranges it names. The top-right corner holds no samples and no range polygon.
    handles = [Line2D([], [], marker="o", linestyle="none",
                      markerfacecolor=d["color_by_pop"][population],
                      markeredgecolor="black", markeredgewidth=0.3,
                      markersize=5, label=POPULATION_LABELS[population])
               for population in LEGEND_ORDER]
    # Bonobo is a range with no sample behind it, so it is keyed as a filled patch
    # rather than a marker -- the shape says which layer it belongs to.
    if d["ranges"] is not None and BONOBO_POPULATION in set(d["ranges"]["population"]):
        handles.append(Patch(facecolor=d["color_by_pop"][BONOBO_POPULATION],
                             edgecolor="none", alpha=RANGE_ALPHA,
                             label=POPULATION_LABELS[BONOBO_POPULATION]))
    legend = ax.legend(handles=handles, loc=MAP_LEGEND_LOC,
                       fontsize=FONT_SIZES["tick"], handletextpad=0.4,
                       labelspacing=0.22, borderpad=0.35, borderaxespad=0.3,
                       frameon=True, framealpha=1.0, edgecolor="0.7")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(8)

    for label, (x, y, ha) in BARRIER_LABELS.items():
        ax.annotate(label, xy=(x, y), ha=ha, va="center",
                    fontsize=FONT_SIZES[BARRIER_LABEL_SIZE],
                    color=GAP_COLOR if label == "Dahomey Gap" else RIVER_COLOR,
                    zorder=7)


def panel_b(ax, d, inset=False):
    """
    Genotype PCA of the samples, coloured by subspecies (the hybrid included).

    The coordinates are the Procrustes-aligned ones: the PCA has been rotated,
    reflected and scaled onto the sampling longitudes and latitudes by the prep
    script, so this panel shares the map's frame and its axes are degrees about
    the sample centroid rather than PC units. It is a rotated PCA, not a claim
    about where any animal lives -- the fit is r = 0.90 along longitude and much
    weaker across latitude, which is the point the panel makes.

    As an inset it sits over the map's ocean, so it takes an opaque background and
    a full frame to separate it from the water, with ticks thinned to what a panel
    this size can carry legibly.
    """
    if inset:
        ax.set_facecolor("white")
        ax.patch.set_alpha(1.0)
        ax.set_zorder(5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2, prune="both"))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=2, prune="both"))
        ax.tick_params(labelsize=FONT_SIZES["annotation"], pad=1.5,
                       length=2, width=0.6)

    individuals = d["pop_labels"].drop_duplicates("individual_id")
    aligned = d["pca_aligned"].merge(
        individuals[["individual_id", "population"]], on="individual_id", how="left")

    for population in LEGEND_ORDER:
        points = aligned[aligned["population"] == population]
        if points.empty:
            continue
        ax.scatter(points["PC1_aligned"], points["PC2_aligned"], s=PCA_MARKER,
                   facecolor=d["color_by_pop"][population],
                   edgecolors="none", alpha=PCA_MARKER_ALPHA, zorder=2)

    # Not equal aspect, deliberately. Both axes are degrees after the alignment, so
    # equal aspect is the honest default -- but the fit is very unequal between
    # them (r = 0.90 along longitude against 0.27 across latitude), so an
    # equal-aspect box would spend most of its area showing how poorly the second
    # axis matches. The panel is a rotated PCA read against the map's east-west
    # axis; the per-axis fits are recorded in params.csv and belong in the caption.
    label_size = FONT_SIZES["annotation"] if inset else FONT_SIZES["base"]
    ax.set_xlabel("Aligned PC1 (°)", fontsize=label_size,
                  labelpad=1.5 if inset else None)
    ax.set_ylabel("Aligned PC2 (°)", fontsize=label_size,
                  labelpad=1.5 if inset else None)


def panel_c(ax, d):
    """
    Ne(t) per subspecies: 1/(2*mean lambda) over that subspecies' within-pairs
    (bold), over one thin line per within-pair, 1/(2*lambda_i).

    The bold line is not the average of the thin ones. The rates are averaged
    before the reciprocal -- 1/x is convex, and the rate estimator's floor makes
    any reciprocal-first average diverge -- so the mean sits below the visual
    centre of its bundle by construction.

    Returns the two style handles describing that distinction.
    """
    key_columns = {"population", "individual_1", "individual_2"}
    times = _trace_times(d["ne_pairs"], key_columns)
    for population in POPULATION_ORDER:
        traces = d["ne_pairs"][d["ne_pairs"]["population"] == population]
        if traces.empty:
            continue
        add_trace_bundle(ax, times, traces.drop(columns=key_columns).values,
                         d["color_by_pop"][population])

    mean_times = d["ne_mean"].index.astype(float)
    for population in POPULATION_ORDER:
        if population in d["ne_mean"]:
            ax.plot(mean_times, d["ne_mean"][population],
                    color=d["color_by_pop"][population], **MEAN_STYLE)

    time_axes(ax, d["params"])
    ax.set_ylim(d["params"]["ne_ymin"], d["params"]["ne_ymax"])
    ax.set_xlabel(TIME_LABEL)
    # The factor of 2 is not decorative: compute_effective_sizes returns
    # 1 / (2 * mean(lambda)), the pairwise-coalescent-rate definition of Ne.
    ax.set_ylabel("Effective population size\n"
                  r"$N_e(t) = \dfrac{1}{2\,\mathrm{mean}(\lambda_{\mathrm{within}})}$")

    return [Line2D([], [], color="0.3", label="Population mean", **MEAN_STYLE),
            Line2D([], [], color="0.3", lw=0.6, alpha=0.5, label="Individual pairs")]


def panel_d(ax, d):
    """
    Relative diversity rho_AB(t) = mean between-pair rate / pooled mean within-pair
    rate, for the six pairs among the four subspecies (bold), over one thin line
    per between-pair.

    rho -> 1 where two subspecies coalesce as readily with each other as within
    themselves, so the dashed reference at 1 is where the pair stops being
    distinguishable; rho << 1 at recent times is the separation between them.
    """
    # Only the six means are drawn here. With 1001 individual trajectories the
    # bundles merged into a single band that hid exactly the differences between
    # pairs the panel exists to show; panel C keeps its per-pair lines, where 377
    # traces in four bundles still read.
    mean_times = d["rho_mean"].index.astype(float)
    for pair_label in d["rho_mean"].columns:
        ax.plot(mean_times, d["rho_mean"][pair_label],
                color=d["color_by_pair"][pair_label], **MEAN_STYLE)

    ax.axhline(1.0, color="0.35", lw=0.8, linestyle=":", zorder=3)
    # Half-way reference, and where each pair reaches it. The verticals are drawn
    # in their pair's own colour and stop at the crossing, so each one reads as
    # belonging to its curve rather than as a gridline across the panel.
    ax.axhline(RHO_CROSSING_LEVEL, color="0.35", lw=0.8, linestyle="--", zorder=3)
    # Full-height verticals, so each crossing can be read straight off the years
    # axis on top as well as the generations axis below. Three of the six nearly
    # coincide -- the NC pairs fall within 6% of each other and the two Western
    # pairs within 2% -- so they overprint; that is a real feature of the data
    # rather than something to space out, and the legend gives the colours.
    for _, row in d["crossings"].iterrows():
        ax.axvline(row["crossing_generations"],
                   color=d["color_by_pair"][row["pair_label"]],
                   lw=0.9, linestyle="--", zorder=2)

    time_axes(ax, d["params"])
    ax.set_ylim(d["params"]["rho_ymin"], d["params"]["rho_ymax"])
    ax.set_xlabel(TIME_LABEL)
    # A ratio of means, not a mean of ratios: compute_relative_divergence averages
    # the between- and within-pair rates separately and then divides.
    ax.set_ylabel("Relative cross-coalescence rate\n"
                  r"$\mathrm{mean}(\lambda_{\mathrm{between}})\,/\,"
                  r"\mathrm{mean}(\lambda_{\mathrm{within}})$")

    # Keyed in the panel rather than labelled on the curves: two pairs sit at the
    # same height at the recent end, so in-place labels collided and the lines ran
    # through the text. Ordered by when each pair reaches rho = 0.5, most recent
    # first, so the legend reads as a ranking of how deep each separation runs and
    # carries the crossing time with it.
    crossing_of = dict(zip(d["crossings"]["pair_label"],
                           d["crossings"]["crossing_years"]))

    def entry(pair_label):
        """One legend handle, carrying the pair's rho = 0.5 crossing time."""
        members = pair_label.split("-")
        # The three NC pairs are named NC-first, so the column reads as a set of
        # comparisons against one population rather than a mixed list.
        if "NC" in members:
            members = ["NC"] + [p for p in members if p != "NC"]
        short = "–".join(PAIR_SHORT[p] for p in members)
        return Line2D([], [], color=d["color_by_pair"][pair_label],
                      label="{}  ($t_{{0.5}}$ = {:.0f} kya)".format(
                          short, crossing_of[pair_label] / 1000.0),
                      **MEAN_STYLE)

    # One column per colour family, each ordered by crossing time. Matplotlib
    # fills the columns top to bottom, so the shorter families are padded with
    # invisible entries to keep the three columns aligned.
    blank = Line2D([], [], linestyle="none", label="")
    columns = [[entry(p) for p in family] for family in PAIR_LEGEND_COLUMNS]
    rows = max(len(c) for c in columns)
    handles = [c[i] if i < len(c) else blank for c in columns for i in range(rows)]
    # Drawn by assemble() in the strip beneath the figure, not in the panel: the
    # entries carry the crossing times now and are too wide to sit over the curves.
    return handles


# ---------------------------------------------------------------------------
# Assembly: owns every inch of the figure.
# ---------------------------------------------------------------------------
def assemble(d):
    fig_w = WIDTHS.get(WIDTH, WIDTH)

    # The map spans the full content width at equal aspect, so its height follows
    # from its own bounding box rather than being chosen.
    lon_span = d["params"]["map_lon_max"] - d["params"]["map_lon_min"]
    lat_span = d["params"]["map_lat_max"] - d["params"]["map_lat_min"]
    map_w = fig_w - MARGIN_L - MARGIN_R
    map_h = map_w * lat_span / lon_span

    # Both bottom panels carry a y axis, so each loses a MARGIN_L to its labels.
    square = (fig_w - MARGIN_L - MARGIN_R - COL_GAP - MARGIN_L) / 2

    fig_h = (MARGIN_T + map_h + MARGIN_B + ROW_GAP
             + square + MARGIN_B + LEGEND_H)
    fig = new_figure(fig_w, fig_h)

    bottom_y = MARGIN_B + LEGEND_H
    top_y = bottom_y + square + MARGIN_B + ROW_GAP

    ax_a = add_axes_inches(fig, MARGIN_L, top_y, map_w, map_h)
    ax_c = add_axes_inches(fig, MARGIN_L, bottom_y, square, square)
    ax_d = add_axes_inches(fig, MARGIN_L + square + COL_GAP + MARGIN_L,
                           bottom_y, square, square)

    panel_a(ax_a, d)

    # The PCA sits in the map's Atlantic. Its box is in map coordinates, so it is
    # converted to inches through the map's own axes: the inset then keeps its
    # place over the ocean whatever the figure width or the map bounds become.
    box = PCA_INSET_BOX
    x0 = MARGIN_L + map_w * (box["lon_min"] - d["params"]["map_lon_min"]) / lon_span
    x1 = MARGIN_L + map_w * (box["lon_max"] - d["params"]["map_lon_min"]) / lon_span
    y0 = top_y + map_h * (box["lat_min"] - d["params"]["map_lat_min"]) / lat_span
    y1 = top_y + map_h * (box["lat_max"] - d["params"]["map_lat_min"]) / lat_span
    # The inset's own labels and ticks live inside the box, so the axes are inset
    # from its bottom and left edges and the whole thing stays over open water.
    label_strip = (y1 - y0) * PCA_INSET_LABEL_PAD
    left_strip = (x1 - x0) * PCA_INSET_LEFT_PAD
    ax_b = add_axes_inches(fig, x0 + left_strip, y0 + label_strip,
                           (x1 - x0) - left_strip, (y1 - y0) - label_strip)

    panel_b(ax_b, d, inset=True)
    style_handles = panel_c(ax_c, d)
    pair_handles = panel_d(ax_d, d)

    # Colour means subspecies in A, B and C, so that key is shared at the foot of
    # the figure. Panel C's line weight means something only there, so its key
    # stays inside it, in grey, to keep the two encodings apart.
    ax_c.legend(handles=style_handles, loc="lower left",
                fontsize=FONT_SIZES["annotation"], handlelength=1.8,
                handletextpad=0.5, labelspacing=0.25, borderaxespad=0.3)

    add_panel_label(ax_a, "A", dx=-26, dy=2)
    # B labels the inset from just above its frame: inside the axes it would sit on
    # the scatter, and out in the figure margin it would land on the map's ocean.
    add_panel_label(ax_b, "B", dx=-20, dy=9)
    add_panel_label(ax_c, "C", dx=-26, dy=2)
    add_panel_label(ax_d, "D", dx=-26, dy=2)

    # Centred under panel D, on the axes' own midpoint rather than the figure's,
    # so the key sits beneath the panel whose colours it explains.
    panel_d_centre = ax_d.get_position().x0 + ax_d.get_position().width / 2
    fig.legend(handles=pair_handles, loc="center", ncol=LEGEND_NCOL,
               bbox_to_anchor=(panel_d_centre, LEGEND_H / 2 / fig_h),
               bbox_transform=fig.transFigure,
               fontsize=FONT_SIZES["base"], handlelength=1.0,
               handletextpad=0.25, columnspacing=0.6)
    return fig


#%%
if __name__ == "__main__":
    save_figure(assemble(load_data()), FIGURE, KIND)
# %%
