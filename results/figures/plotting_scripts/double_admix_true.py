#%%
"""
The two-admixture example under true local ancestry: the demography (A), the
whole-genome embeddings (B), how well their reconstructed rates match the input
ones (C), and one embeddings panel per source ancestry (D).

Data: results/figure_data/double_admix_true/, written by
utils/runs/prepare_figure_data/double_admix_true.py -- the ancestry embeddings
there are already Procrustes-aligned.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
import demes
import demesdraw
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils/snakemake_utils")
from PlotEmbeddings import *
from evaluate_performance import reconstructed_vs_input_hist2d
plt.rcParams['savefig.bbox'] = None

FIGURE = "double_admix_true"
graph = demes.load("../../../data/demes/two_admix.yaml")

pop_colors = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
pop_labels = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))

embeddings_WG = pd.read_parquet(figure_data(FIGURE, "embeddings_WG.parquet"))
input_rates_WG = pd.read_parquet(figure_data(FIGURE, "input_rates_WG.parquet"))
reconstructed_rates_WG = pd.read_parquet(figure_data(FIGURE, "reconstructed_rates_WG.parquet"))
ancestry_embeddings = pd.read_parquet(figure_data(FIGURE, "ancestry_embeddings.parquet"))
params = pd.read_csv(figure_data(FIGURE, "params.csv")).set_index("parameter")["value"]

min_time = float(params["min_time"])
max_time = float(params["max_time"])
xy_lims = float(params["xy_lims"])
N_max = float(params["N_max"])

ANCESTRIES = ["A", "B", "C", "D"]
true_embeddings = {a: ancestry_embeddings[ancestry_embeddings["ancestry"] == a]
                   for a in ANCESTRIES}

plotter = PlotEmbeddings(embeddings_df = embeddings_WG,labels_df = pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
RENDER_KW = dict(width = 600, height = 600, scale = 4)
img, aspect_ratio = plotter.save_static(**RENDER_KW)
# %%
W = 6.5
grid_w_gap = 0
grid_emebeddings_w = (W - 3*grid_w_gap)/4
grid_embeddings_h = grid_emebeddings_w / aspect_ratio

row_w_gap = 0
WG_embeddings_h = (W-2*row_w_gap)/(2+aspect_ratio)
WG_embeddings_w = WG_embeddings_h * aspect_ratio

row_hgap = 0.6
H = WG_embeddings_h + grid_embeddings_h + row_hgap
# GridSpec hspace is a fraction of the mean row height, not inches; convert so that
# row_hgap is honored as a true inch gap and the rows keep their intended inch heights
outer_hspace = row_hgap / ((WG_embeddings_h + grid_embeddings_h) / 2)
#%%
fig = plt.figure(figsize=(W, H), dpi=500,constrained_layout=False)
# margins pinned to the full figure so the height/width ratios below map to real inches
outer = gridspec.GridSpec(
    nrows=2, ncols=1, height_ratios=[WG_embeddings_h, grid_embeddings_h],
    hspace=outer_hspace, left=0, right=1, top=1, bottom=0, figure=fig)

# ---- top row: demography schematic + whole-genome embeddings + input-vs-reconstructed rates ----
top = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=3, subplot_spec=outer[0],
    width_ratios=[WG_embeddings_h, WG_embeddings_w, WG_embeddings_h], wspace=row_w_gap)
demography_ax = fig.add_subplot(top[0, 0])
WG_embeddings_ax = fig.add_subplot(top[0, 1])
rates_hist_ax = fig.add_subplot(top[0, 2])

# ---- bottom: 1x4 row of true local-ancestry embeddings ----
bottom = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=4, subplot_spec=outer[1],
    width_ratios=[grid_emebeddings_w]*4, height_ratios=[grid_embeddings_h],
    wspace=grid_w_gap, hspace=0,
)
ancestry_axes = [[fig.add_subplot(bottom[0, j]) for j in range(4)]]
#%%
# First row
demesdraw.tubes(graph, ax=demography_ax, colours=pop_colors_dict, labels = "xticks", max_time = max_time, log_time = True)
demography_ax.set_ylim(bottom=min_time)
demography_ax.set_ylabel("Time (generations ago)", fontsize=10)
demography_ax.tick_params(axis="x", labelsize=7)
reconstructed_vs_input_hist2d(input_rates_WG, reconstructed_rates_WG, N_max, ax=rates_hist_ax, x_label="Input Rates", draw_colorbar=False)
rates_hist_ax.yaxis.tick_right()
rates_hist_ax.yaxis.set_label_position("right")
rates_hist_ax.yaxis.set_label_coords(1.15, 0.4)
rates_hist_ax.xaxis.set_label_coords(0.5, -0.1)
rates_hist_ax.tick_params(labelsize=6.5)
rates_hist_ax.set_xlabel(r"Input Rates + $\frac{1}{N_{\max}}$", fontsize=9)
rates_hist_ax.set_ylabel(r'Reconstructed Rates + $\frac{1}{N_{\max}}$', fontsize=9)
WG_embeddings_ax.spines['left'].set_visible(False)
WG_embeddings_ax.spines['bottom'].set_visible(False)
WG_embeddings_ax.set_xticks([])
WG_embeddings_ax.set_yticks([])
WG_embeddings_ax.imshow(img)
pos = WG_embeddings_ax.get_position()
xnudge = 0
ynudge = 0.04
new_pos = [pos.x0 - xnudge, pos.y0 - ynudge, pos.width, pos.height]
WG_embeddings_ax.set_position(new_pos)
#%%
# Lower grid: one panel per source ancestry, in A,B,C,D order
for j, ancestry in enumerate(ANCESTRIES):
    plotter = PlotEmbeddings(embeddings_df = true_embeddings[ancestry], labels_df = pop_labels,
                             colors_df = pop_colors, sample_id_col = "individual_id")
    plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
    plotter.add_sample_trajectories()
    img_ancestry, _ = plotter.save_static(**RENDER_KW)
    ax = ancestry_axes[0][j]
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(img_ancestry, aspect = "equal")

#%%
# Panel labels, anchored to the top-left of each labelled axis/row
row1_top = demography_ax.get_position().y1
row2_top = ancestry_axes[0][0].get_position().y1

fig.text(demography_ax.get_position().x0-0.03, row1_top, "A", fontsize=12, fontweight="bold", va="top", ha="left")
fig.text(WG_embeddings_ax.get_position().x0, row1_top, "B", fontsize=12, fontweight="bold", va="top", ha="left")
fig.text(rates_hist_ax.get_position().x0-0.03, row1_top, "C", fontsize=12, fontweight="bold", va="top", ha="left")

fig.text(ancestry_axes[0][0].get_position().x0-0.01, row2_top, "D", fontsize=12, fontweight="bold", va="top", ha="left")

for j, ancestry in enumerate(ANCESTRIES):
    ancestry_axes[0][j].set_title(f"{ancestry} Ancestry", fontsize=10)

filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../sim_figs/{filename}.pdf", dpi=500, bbox_inches="tight", pad_inches=0.01)
# %%
