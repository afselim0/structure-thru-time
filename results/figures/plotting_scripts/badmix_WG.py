#%%
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
sys.path.append("../../../utils/snakemake_utils")
sys.path.append("../../../utils/misc_utils")
from PlotEmbeddings import *
from OptimizationFunctions import procrustes_align_trajectories_df

misc_plot_path = "../sim_figs_png/misc"
three_way_graph = demes.load("../../../data/demes/three_way_admixture.yaml")
ghost_graph = demes.load("../../../data/demes/ghost_admixture.yaml")
bottleneck_graph = demes.load("../../../data/demes/recent_bottleneck.yaml")

three_way_pop_labels = pd.read_csv("../../outputs/simulations/three_way_admixture/sim/data/pop_labels.csv")
ghost_pop_labels = pd.read_csv("../../outputs/simulations/ghost_admixture/sim/data/pop_labels.csv")
bottleneck_pop_labels = pd.read_csv("../../outputs/simulations/recent_bottleneck/sim/data/pop_labels.csv")

# Each simulation's own pop_colors.csv assigns the palette in a different
# population order (e.g. A is purple in three_way/ghost but green in
# recent_bottleneck), so a population would change color between panels. Pin one
# canonical population -> color map, in A,B,C,D order, shared by all examples
pop_colors = pd.DataFrame({
    "population": ["A", "B", "C", "D"],
    "color": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
})
pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
min_time = 5
max_time = 5000
xy_lims = 1e6

three_way_embeddings_WG = pd.read_parquet("../../outputs/simulations/three_way_admixture/sim/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")
ghost_embeddings_WG = pd.read_parquet("../../outputs/simulations/ghost_admixture/sim/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")
bottleneck_embeddings_WG = pd.read_parquet("../../outputs/simulations/recent_bottleneck/sim/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS20.maxWS300.V1.FS1.0.maxT5000.nT2500.parquet")

three_way_embeddings_WG = procrustes_align_trajectories_df(bottleneck_embeddings_WG, three_way_embeddings_WG)
ghost_embeddings_WG = procrustes_align_trajectories_df(bottleneck_embeddings_WG, ghost_embeddings_WG)
#%%
plotter = PlotEmbeddings(embeddings_df = three_way_embeddings_WG, labels_df = three_way_pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
# Every embeddings panel is rendered on this canvas. The cropped aspect ratio
# depends on the canvas pixel size (Plotly lays text out in fixed pixels), so the
# probe below and the final renders must share it for the panels to fill their axes
RENDER_KW = dict(width = 1475, height = 1475)
_, y = plotter.save_static(**RENDER_KW, scale = 1)  # cheap probe: scale does not affect the ratio
# y=0.8
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
demesdraw.tubes(three_way_graph, ax = axs[0,0], colours = pop_colors_dict, log_time = True, max_time = max_time, labels = "xticks")
demesdraw.tubes(ghost_graph, ax = axs[0,1], colours = pop_colors_dict, log_time = True, max_time = max_time, labels = "xticks")
demesdraw.tubes(bottleneck_graph, ax = axs[0,2], colours = pop_colors_dict, log_time = True, max_time = max_time, labels = "xticks")
axs[0,0].set_ylim(bottom=min_time)
axs[0,1].set_ylim(bottom=min_time)
axs[0,2].set_ylim(bottom=min_time)
axs[0,0].set_ylabel("Time (generations ago)", fontsize = 11)
axs[0,1].set_ylabel("Time (generations ago)", fontsize = 11)
axs[0,2].set_ylabel("Time (generations ago)", fontsize = 11)

# 3d plots
plotter = PlotEmbeddings(embeddings_df = three_way_embeddings_WG, labels_df = three_way_pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = 50, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
img.save(f"{misc_plot_path}_three_way.png")
axs[1,0].set_axis_off()
axs[1,0].imshow(img, aspect = "equal")

plotter = PlotEmbeddings(embeddings_df = ghost_embeddings_WG, labels_df = ghost_pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = 50, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
img.save(f"{misc_plot_path}_ghost.png")
axs[1,1].set_axis_off()
axs[1,1].imshow(img, aspect = "equal")

plotter = PlotEmbeddings(embeddings_df = bottleneck_embeddings_WG, labels_df = bottleneck_pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = 5, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
img.save(f"{misc_plot_path}_bottleneck.png")
axs[1,2].set_axis_off()
axs[1,2].imshow(img, aspect = "equal")

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
