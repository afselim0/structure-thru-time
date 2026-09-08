#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
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

misc_plot_path = "../sim_figs_png/misc_ancestry"
sim_path = "../../outputs/simulations"

# Columns are examples, rows are the ancestries carried by admixed individuals.
# Each example ran the embedding with its own window-size settings, so the
# embeddings filename suffix differs per example
examples = ["three_way_admixture", "ghost_admixture", "recent_bottleneck"]
example_titles = ["Three-way admixture", "Ghost admixture", "Recent bottleneck"]
ancestries = ["B", "C", "D"]
embeddings_suffix = {
    "three_way_admixture": "embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet",
    "ghost_admixture":     "embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet",
    "recent_bottleneck":   "embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].nTe2500.minWS20.maxWS300.V1.FS1.0.maxT5000.nT2500.parquet",
}

# Each simulation's own pop_colors.csv assigns the palette in a different
# population order (e.g. A is purple in three_way/ghost but green in
# recent_bottleneck), so a population would change color between panels. Pin one
# canonical population -> color map, in A,B,C,D order, shared by all examples
pop_colors = pd.DataFrame({
    "population": ["A", "B", "C", "D"],
    "color": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
})

# Labels are per-example: they map each individual to its source population
pop_labels = {ex: pd.read_csv(f"{sim_path}/{ex}/sim/data/pop_labels.csv") for ex in examples}

# recent_bottleneck's structure is recent enough to need a shorter time axis
min_time = {"three_way_admixture": 50, "ghost_admixture": 50, "recent_bottleneck": 10}
max_time = 5000
xy_lims = 1e6

FIG_WIDTH = 6.5
MARGIN_LEFT, MARGIN_RIGHT = 0.15, 0.15
MARGIN_TOP, MARGIN_BOTTOM = 0.25, 0.35  # bottom margin holds the population color legend
wspace, hspace = 0, 0.05  # fractions of a panel's width/height, as matplotlib defines them


def load_ancestry_embeddings(method: str) -> dict:
    """Load the per-ancestry embeddings for one LAI method, keyed by (example, ancestry)."""
    embeddings = {}
    for ex in examples:
        for anc in ancestries:
            path = f"{sim_path}/{ex}/sim/ancestry_STT/{method}/{anc}/data/{embeddings_suffix[ex]}"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {method} embeddings for {ex}/{anc}: {path}")
            embeddings[(ex, anc)] = pd.read_parquet(path)
    return embeddings


def make_plotter(embeddings_df: pd.DataFrame, example: str) -> PlotEmbeddings:
    """Build an initialized plotter with trajectories for one panel."""
    plotter = PlotEmbeddings(
        embeddings_df=embeddings_df,
        labels_df=pop_labels[example],
        colors_df=pop_colors,
        sample_id_col="individual_id",
    )
    plotter.init_plot(min_time=min_time[example], max_time=max_time, time_scale="log", xy_lims=xy_lims)
    plotter.add_sample_trajectories()
    return plotter


def build_ancestry_figure(method: str) -> plt.Figure:
    """
    Build the 3x3 figure of per-ancestry embeddings for one LAI method:
    rows are ancestries (B, C, D), columns are examples.
    """
    embeddings = load_ancestry_embeddings(method)

    # Procrustes-align each panel to the same reference so that trajectories are
    # comparable across ancestries and examples. The reference is the first panel
    # (top-left), which is left unrotated
    reference = embeddings[(examples[1], ancestries[0])]
    for key, df in embeddings.items():
        if key != (examples[0], ancestries[0]):
            embeddings[key] = procrustes_align_trajectories_df(reference, df)

    # The static export's aspect ratio is only knowable after a render, so probe
    # the first panel and derive every panel's height from it. The cropped ratio
    # depends on the canvas pixel size, so every panel is rendered at RENDER_KW --
    # the probe included (scale is safe to lower, it does not affect the ratio)
    RENDER_KW = dict(width=1475, height=1475)
    _, aspect_ratio = make_plotter(reference, examples[0]).save_static(**RENDER_KW, scale=1)
    col_width = (FIG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / (3 + 2*wspace)
    row_height = col_width / aspect_ratio
    fig_height = MARGIN_TOP + MARGIN_BOTTOM + row_height*(3 + 2*hspace)

    fig = plt.figure(figsize=(FIG_WIDTH, fig_height), dpi=500)
    # Margins are pinned explicitly in figure fractions: matplotlib's defaults
    # (~top=0.88, bottom=0.11, left=0.125, right=0.9) would otherwise silently eat
    # ~20-25% of the figure and shrink every panel below its computed size
    gs = gridspec.GridSpec(
        nrows=3, ncols=3, figure=fig,
        left=MARGIN_LEFT/FIG_WIDTH, right=1 - MARGIN_RIGHT/FIG_WIDTH,
        bottom=MARGIN_BOTTOM/fig_height, top=1 - MARGIN_TOP/fig_height,
        wspace=wspace, hspace=hspace,
    )
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)])

    # Render each panel into its axis, at the same canvas size as the probe above
    for i, anc in enumerate(ancestries):
        for j, ex in enumerate(examples):
            plotter = make_plotter(embeddings[(ex, anc)], ex)
            img, _ = plotter.save_static(**RENDER_KW)
            img.save(f"{misc_plot_path}_{method}_{ex}_{anc}.png")
            axs[i, j].set_axis_off()
            axs[i, j].imshow(img, aspect="equal")

    # Column titles (examples) and row labels (ancestries), placed against each
    # panel's gridspec position
    for j, title in enumerate(example_titles):
        pos = axs[0, j].get_position()
        fig.text(pos.x0 + pos.width/2, 1 - 0.06*MARGIN_TOP, title,
                 ha="center", va="top", fontsize=9)
    for i, anc in enumerate(ancestries):
        pos = axs[i, 0].get_position()
        fig.text(0.012, pos.y0 + pos.height/2, f"{anc} ancestry",
                 ha="center", va="center", rotation=90, fontsize=9)

    # Shared source-population legend, centered in the reserved bottom margin.
    # Proxy handles carry the canonical colors, since the panels are rasterized
    # Plotly exports with no matplotlib artists to build a legend from
    handles = [Line2D([], [], color=row["color"], lw=2, label=row["population"])
               for _, row in pop_colors.iterrows()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, handlelength=1.2,
               columnspacing=1.5, bbox_to_anchor=(0.5, 0))

    return fig


# %%
gnn_fig = build_ancestry_figure("GNN")
gnn_fig.savefig("../sim_figs/badmix_GNN.pdf")

# %%
rfmix_fig = build_ancestry_figure("RFmix")
rfmix_fig.savefig("../sim_figs/badmix_RFmix.pdf")
# %%
