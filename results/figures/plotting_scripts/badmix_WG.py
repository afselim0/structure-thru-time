#%%
"""
Three admixture scenarios that are hard for the method: three-way admixture, ghost
admixture and a recent bottleneck. Top row the demographies (A), bottom row the
whole-genome embeddings (B).

Data: results/figure_data/badmix_WG/, written by
utils/runs/prepare_figure_data/badmix_WG.py -- the embeddings there are already
Procrustes-aligned to the recent_bottleneck frame.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
import demes
import demesdraw
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
from PlotEmbeddings import *

misc_plot_path = "../sim_figs_png/misc"
FIGURE = "badmix_WG"

EXAMPLES = ["three_way_admixture", "ghost_admixture", "recent_bottleneck"]
graphs = {ex: demes.load(f"../../../data/demes/{ex}.yaml") for ex in EXAMPLES}

all_embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
all_pop_labels = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
pop_colors = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
params = pd.read_csv(figure_data(FIGURE, "params.csv")).set_index("parameter")["value"]

pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
min_time = float(params["min_time"])
max_time = float(params["max_time"])
xy_lims = float(params["xy_lims"])

embeddings = {ex: all_embeddings[all_embeddings["example"] == ex] for ex in EXAMPLES}
pop_labels = {ex: all_pop_labels[all_pop_labels["example"] == ex] for ex in EXAMPLES}
# the embeddings panels start at 50 generations except the bottleneck, whose structure
# is recent enough to need the shorter axis
panel_min_time = {"three_way_admixture": 50, "ghost_admixture": 50,
                  "recent_bottleneck": min_time}
#%%
plotter = PlotEmbeddings(embeddings_df = embeddings[EXAMPLES[0]], labels_df = pop_labels[EXAMPLES[0]],
                         colors_df = pop_colors, sample_id_col = "individual_id")
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
# Every embeddings panel is rendered on this canvas. The cropped aspect ratio
# depends on the canvas pixel size (Plotly lays text out in fixed pixels), so the
# probe below and the final renders must share it for the panels to fill their axes
RENDER_KW = dict(width = 1475, height = 1475)
_, y = plotter.save_static(**RENDER_KW, scale = 1)  # cheap probe: scale does not affect the ratio
# %%
FIG_WIDTH = 6.5
# Margins reserved around the grid, in inches. Left margin holds the demesdraw
# y-axis labels; the rest keep tick labels from being clipped at the figure edge
MARGIN_LEFT, MARGIN_RIGHT = 0.5, 0.1
MARGIN_TOP, MARGIN_BOTTOM = 0.1, 0.35
wspace, hspace = 0, 0.2  # fractions of a panel's width/height, as matplotlib defines them

# Each embeddings panel must be col_width x (col_width / aspect_ratio) inches to
# fill its axis exactly, so derive the row height from the true column width
col_width = (FIG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT) / (3 + 2*wspace)
row_height = col_width / y
FIG_HEIGHT = MARGIN_TOP + MARGIN_BOTTOM + row_height*(2 + hspace)

fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=500)
# Margins are pinned explicitly in figure fractions: matplotlib's defaults
# (~top=0.88, bottom=0.11, left=0.125, right=0.9) would otherwise silently eat
# ~20-25% of the figure and shrink every panel below its computed size
gs = gridspec.GridSpec(
    nrows=2, ncols=3, figure=fig,
    left=MARGIN_LEFT/FIG_WIDTH, right=1 - MARGIN_RIGHT/FIG_WIDTH,
    bottom=MARGIN_BOTTOM/FIG_HEIGHT, top=1 - MARGIN_TOP/FIG_HEIGHT,
    wspace=wspace, hspace=hspace,
)
axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])

# Plotting demes
for j, ex in enumerate(EXAMPLES):
    demesdraw.tubes(graphs[ex], ax = axs[0, j], colours = pop_colors_dict,
                    log_time = True, max_time = max_time, labels = "xticks")
    axs[0, j].set_ylim(bottom=min_time)
    axs[0, j].set_ylabel("Time (generations ago)", fontsize = 11)

# 3d plots
for j, ex in enumerate(EXAMPLES):
    plotter = PlotEmbeddings(embeddings_df = embeddings[ex], labels_df = pop_labels[ex],
                             colors_df = pop_colors, sample_id_col = "individual_id")
    plotter.init_plot(min_time = panel_min_time[ex], max_time = max_time,
                      time_scale = "log", xy_lims = xy_lims)
    plotter.add_sample_trajectories()
    img, _ = plotter.save_static(**RENDER_KW)
    img.save(f"{misc_plot_path}_{ex}.png")
    axs[1, j].set_axis_off()
    axs[1, j].imshow(img, aspect = "equal")

# Nudge the flanking embeddings panels outward to widen the gaps between the
# three cubes
def nudge_ax_x(ax, shift_in):
    pos = ax.get_position()
    ax.set_position([pos.x0 + shift_in/FIG_WIDTH, pos.y0, pos.width, pos.height])

nudge_ax_x(axs[1,0], -0.2)
nudge_ax_x(axs[1,2], 0.2)

# Panel labels, one per row, anchored to the top-left of the figure
fig.text(0.01, 1, "A", fontsize = 12, fontweight = "bold", va = "top")
fig.text(0.01, 0.5, "B", fontsize = 12, fontweight = "bold", va = "top")

filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../sim_figs/{filename}.pdf")
# %%
