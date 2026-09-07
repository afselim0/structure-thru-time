#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from copy import copy
import numpy as np
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils/snakemake_utils")
sys.path.append("../../../utils/misc_utils")
from PlotEmbeddings import *
from OptimizationFunctions import procrustes_align_trajectories_df
from evaluate_performance import reconstructed_vs_input_hist2d
plt.rcParams['savefig.bbox'] = None

sim_path = "../../outputs/simulations/two_admix/sim"
EMB_SUFFIX = "embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet"
RATES_SUFFIX = "rates.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet"
RECON_SUFFIX = "reconstructed_rates.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet"

ancestries = ["A", "B", "C", "D"]
methods = ["GNN", "RFmix"]

# Ancestries A,B admix into Admix_1 and C,D into Admix_2 (see data/demes/two_admix.yaml),
# so each column's rate panel only shows that column's admixed population against everyone else
col_admix_pop = {"A": "Admix_1", "B": "Admix_1", "C": "Admix_2", "D": "Admix_2"}

# Load embeddings, input rates and reconstructed rates for every method/ancestry combination
embeddings = {m: {a: pd.read_parquet(f"{sim_path}/ancestry_STT/{m}/{a}/data/{EMB_SUFFIX}") for a in ancestries} for m in methods}
input_rates = {m: {a: pd.read_parquet(f"{sim_path}/ancestry_STT/{m}/{a}/data/{RATES_SUFFIX}") for a in ancestries} for m in methods}
reconstructed_rates = {m: {a: pd.read_parquet(f"{sim_path}/ancestry_STT/{m}/{a}/data/{RECON_SUFFIX}") for a in ancestries} for m in methods}

pop_labels = pd.read_csv(f"{sim_path}/data/pop_labels.csv")
pop_colors = pd.read_csv(f"{sim_path}/data/pop_colors.csv")
pop_colors.iloc[-1, -1] = "#505011"
pop_colors_dict = dict(zip(pop_colors["population"], pop_colors["color"]))

min_time = 50
max_time = 5000
xy_lims = 1e6
N_max = 1000000
time_scale = "log"
INDIVIDUAL_LINE_WIDTH = 0.2
INDIVIDUAL_LINE_ALPHA = 0.4
MEAN_LINE_WIDTH = 1.5
MEAN_LINE_ALPHA = 0.9

# Align each method's per-ancestry embeddings to a common reference frame so panels are visually comparable;
# A/B share the Admix_1 frame and C/D share the Admix_2 frame
for method in methods:
    print(f"Aligning {method}-ancestry embeddings to reference...")
    ref_AB = embeddings[method]["A"].copy()
    ref_CD = embeddings[method]["D"].copy()
    for ancestry in ancestries:
        ref = ref_AB if ancestry in ("A", "B") else ref_CD
        embeddings[method][ancestry] = procrustes_align_trajectories_df(ref, embeddings[method][ancestry], allow_reflections=True)

#%%
def method_rate_lims(method: str) -> tuple:
    """
    Common log-log axis limits for one method, spanning every ancestry's input
    (shifted by 1/N_max) and reconstructed rates. Shared by the rate panels and
    the histogram panels so both rows sit on exactly the same rate scale.
    """
    lo, hi = np.inf, -np.inf
    for ancestry in ancestries:
        x = (input_rates[method][ancestry] + 1 / N_max).values.ravel()
        y = reconstructed_rates[method][ancestry].values.ravel()
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        lo = min(lo, x[valid].min(), y[valid].min())
        hi = max(hi, x[valid].max(), y[valid].max())
    return lo, hi


rate_lims = {m: method_rate_lims(m) for m in methods}

# Render one embeddings panel first to get the aspect ratio that sets the figure geometry
RENDER_KW = dict(width=600, height=600, scale=4)


def render_embeddings(embeddings_df: pd.DataFrame):
    """Render a static 3D embeddings image for one ancestry."""
    plotter = PlotEmbeddings(embeddings_df=embeddings_df, labels_df=pop_labels,
                             colors_df=pop_colors, sample_id_col="individual_id")
    plotter.init_plot(min_time=min_time, max_time=max_time, time_scale=time_scale, xy_lims=xy_lims)
    plotter.add_sample_trajectories()
    return plotter.save_static(**RENDER_KW)


_, aspect_ratio = render_embeddings(embeddings["GNN"]["A"])

W = 6.5
# embeddings panels stay flush; the rate and histogram panels get a gap so their
# y tick labels do not collide with the neighbouring panel (fraction of panel width)
emb_wspace = 0.0
plot_wspace = 0.35
panel_w = W / 4
panel_h = panel_w / aspect_ratio
# gap between rows 2 and 3 so row 2's x tick labels and axis label have room; rows 1-2 stay flush.
# Nested GridSpecs keep the gap an exact inch value while all three rows keep height panel_h
row_hgap = 0.45
H = 3 * panel_h + row_hgap
outer_hspace = row_hgap / ((2 * panel_h + panel_h) / 2)


#%%
def plot_admix_rates(rates_df: pd.DataFrame, ax: plt.Axes, ancestry: str, lims: tuple) -> None:
    """
    Plot coalescence rates involving this column's admixed individuals under the
    given ancestry: every admixed-vs-other pair colored by the partner
    population, plus the within-admixed pairs colored by the admixed population
    itself. Individual pairs are thin lines and each population's mean is a
    thick line. Rates are shifted by 1/N_max and drawn on the same log scale as
    the input-vs-reconstructed panels below.
    """
    admix_pop = col_admix_pop[ancestry]
    active_samples = np.unique([s for pair in rates_df.index.values for s in pair])
    row_labels = pop_labels[pop_labels["individual_id"].isin(active_samples)]
    times = rates_df.columns.astype(float)

    admix_samples = np.unique(row_labels[row_labels["population"] == admix_pop]["individual_id"].values)
    # the admixed population is plotted last so its within-population rates sit on top
    partner_pops = [p for p in row_labels["population"].unique() if p != admix_pop] + [admix_pop]

    for partner_pop in partner_pops:
        partner_samples = np.unique(row_labels[row_labels["population"] == partner_pop]["individual_id"].values)
        # rates are stored on sorted sample pairs, so canonicalise each pair before lookup
        sample_pairs = [(s1, s2) if s1 < s2 else (s2, s1)
                        for s1 in admix_samples for s2 in partner_samples if s1 != s2]
        sample_pairs = [p for p in dict.fromkeys(sample_pairs) if p in rates_df.index]
        if not sample_pairs:
            continue
        pair_rates = rates_df.loc[sample_pairs, :].values + 1 / N_max
        for row in pair_rates:
            ax.plot(times, row, color=pop_colors_dict[partner_pop],
                    lw=INDIVIDUAL_LINE_WIDTH, alpha=INDIVIDUAL_LINE_ALPHA, rasterized=True)
        ax.plot(times, pair_rates.mean(axis=0), color=pop_colors_dict[partner_pop],
                lw=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA)

    ax.set_xscale(time_scale)
    ax.set_xlim(min_time, max_time)
    ax.set_yscale("log")
    ax.set_ylim(lims)


def build_figure(method: str) -> plt.Figure:
    """
    Build the 3x4 figure for one local-ancestry method: columns are the four
    ancestries, rows are 3D embeddings / admixed-vs-others rates / input-vs-
    reconstructed rate histograms.
    """
    lims = rate_lims[method]

    fig = plt.figure(figsize=(W, H), dpi=500, constrained_layout=False)
    # outer split puts the inch gap between the top two rows and the histogram row
    outer = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2 * panel_h, panel_h],
                              hspace=outer_hspace, left=0, right=1, top=1, bottom=0, figure=fig)
    # rows 1 and 2 are nested separately so the embeddings can stay flush while the rate panels are spaced
    top = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=outer[0],
                                           height_ratios=[panel_h] * 2, hspace=0)
    emb_grid = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=top[0],
                                                width_ratios=[panel_w] * 4, wspace=emb_wspace)
    rate_grid = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=top[1],
                                                 width_ratios=[panel_w] * 4, wspace=plot_wspace)
    bottom = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=4, subplot_spec=outer[1],
                                              width_ratios=[panel_w] * 4, wspace=plot_wspace)

    emb_axes = [fig.add_subplot(emb_grid[0, j]) for j in range(4)]
    # rate panels and histogram panels each share x and y within their row
    rate_axes, hist_axes = [], []
    for j in range(4):
        rate_axes.append(fig.add_subplot(rate_grid[0, j], sharex=rate_axes[0] if rate_axes else None,
                                         sharey=rate_axes[0] if rate_axes else None))
        hist_axes.append(fig.add_subplot(bottom[0, j], sharex=hist_axes[0] if hist_axes else None,
                                         sharey=hist_axes[0] if hist_axes else None))

    # ---- row 1: 3D embeddings ----
    for ax, ancestry in zip(emb_axes, ancestries):
        img, _ = render_embeddings(embeddings[method][ancestry])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img, aspect="equal")
        ax.set_title(f"{ancestry} Ancestry", fontsize=10)

    # ---- row 2: admixed-vs-everyone-else coalescence rates ----
    for ax, ancestry in zip(rate_axes, ancestries):
        plot_admix_rates(input_rates[method][ancestry], ax, ancestry, lims)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Time (generations ago)", fontsize=8)
    rate_axes[0].set_ylabel(r"Rates + $\frac{1}{N_{\max}}$", fontsize=8)
    for ax in rate_axes[1:]:
        ax.tick_params(labelleft=False)

    # ---- row 3: input vs reconstructed rates, on the same shared rate limits as row 2 ----
    for j, (ax, ancestry) in enumerate(zip(hist_axes, ancestries)):
        reconstructed_vs_input_hist2d(input_rates[method][ancestry], reconstructed_rates[method][ancestry],
                                      N_max, ax=ax, x_label="Input Rates", draw_colorbar=False, ylabel=False,
                                      shared_lims=lims, stats_fontsize=4.5)
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel(r"Input Rates + $\frac{1}{N_{\max}}$", fontsize=8)
    hist_axes[0].set_ylabel(r"Reconstructed Rates + $\frac{1}{N_{\max}}$", fontsize=8)
    for ax in hist_axes[1:]:
        ax.tick_params(labelleft=False)

    # Row 2 shares the histograms' rate axis, so copy their tick locators/formatters over verbatim
    rate_axes[0].yaxis.set_major_locator(copy(hist_axes[0].yaxis.get_major_locator()))
    rate_axes[0].yaxis.set_minor_locator(copy(hist_axes[0].yaxis.get_minor_locator()))
    rate_axes[0].yaxis.set_major_formatter(copy(hist_axes[0].yaxis.get_major_formatter()))
    rate_axes[0].yaxis.set_minor_formatter(copy(hist_axes[0].yaxis.get_minor_formatter()))

    # Population legend below the histogram x labels so the two never overlap
    legend_handles = [Line2D([0], [0], color=color, linewidth=3, label=pop)
                      for pop, color in pop_colors_dict.items()]
    fig.legend(handles=legend_handles, title="Population", ncol=len(legend_handles),
               loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, title_fontsize=8, frameon=False)

    # Row labels A/B/C on one vertical line, anchored just left of the embeddings panel so
    # they hug the figure rather than the wider y-axis text of rows 2-3.
    # ha="right" keeps the glyphs entirely left of the anchor, so the offset is a true gap
    label_x = emb_axes[0].get_position().x0 - 0.012

    for ax, letter in zip((emb_axes[0], rate_axes[0], hist_axes[0]), ["A", "B", "C"]):
        fig.text(label_x, ax.get_position().y1 + 0.04, letter, fontsize=12, fontweight="bold",
                 va="top", ha="right")

    return fig


#%%
for method in methods:
    fig = build_figure(method)
    fig.savefig(f"../sim_figs/double_admix_{method}.pdf", dpi=500, bbox_inches="tight", pad_inches=0.01)
    print(f"Saved ../sim_figs/double_admix_{method}.pdf")
# %%
