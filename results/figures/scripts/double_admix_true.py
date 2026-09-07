#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from plotting_style import *
import sys
import os
import demes
import demesdraw
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils")
from PlotEmbeddings import *
from OptimizationFunctions import procrustes_align_trajectories_df
from evaluate_performance import reconstructed_vs_input_hist2d
plt.rcParams['savefig.bbox'] = None

misc_plot_path = "../misc_figs/misc.png"
graph = demes.load("../../../data/demes/main_1/two_admix.yaml")
pop_colors = pd.read_csv("../../simulations/outputs/two_admix/data/pop_colors.csv")
pop_colors.iloc[-1,-1] = "#505011"
pop_labels = pd.read_csv("../../simulations/outputs/two_admix/data/pop_labels.csv")
pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
min_time = 50
max_time = 5000
xy_lims = 1e6
N_max = 1000000

embeddings_WG = pd.read_parquet("../../simulations/outputs/two_admix/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")
input_rates_WG = pd.read_parquet("../../simulations/outputs/two_admix/data/rates.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")
reconstructed_rates_WG = pd.read_parquet("../../simulations/outputs/two_admix/data/reconstructed_rates.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")

true_embeddings_A = pd.read_parquet("../../simulations/outputs/two_admix/ancestry_STT/True/A/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")
true_embeddings_B = pd.read_parquet("../../simulations/outputs/two_admix/ancestry_STT/True/B/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")
true_embeddings_C = pd.read_parquet("../../simulations/outputs/two_admix/ancestry_STT/True/C/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")
true_embeddings_D = pd.read_parquet("../../simulations/outputs/two_admix/ancestry_STT/True/D/data/embeddings.d2.Nmin100.Nmax1000000.p1.linkinverse_distance.L1[None].L2[None].nTe2500.minWS100.maxWS200.V1.FS1.0.maxT5000.nT2500.parquet")

ref_embeddings_AB = true_embeddings_A
ref_embeddings_CD = true_embeddings_D

# Procrustes-align the embeddings to a common reference frame so that they can be compared visually
print("Aligning True-ancestry embeddings to reference...")
true_embeddings_A = procrustes_align_trajectories_df(ref_embeddings_AB, true_embeddings_A, allow_reflections=True)
true_embeddings_B = procrustes_align_trajectories_df(ref_embeddings_AB, true_embeddings_B, allow_reflections=True)
true_embeddings_C = procrustes_align_trajectories_df(ref_embeddings_CD, true_embeddings_C, allow_reflections=True)
true_embeddings_D = procrustes_align_trajectories_df(ref_embeddings_CD, true_embeddings_D, allow_reflections=True)


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
# Lower grid
## True
plotter = PlotEmbeddings(embeddings_df = true_embeddings_A,labels_df = pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
ancestry_axes[0][0].spines['left'].set_visible(False)
ancestry_axes[0][0].spines['bottom'].set_visible(False)
ancestry_axes[0][0].set_xticks([])
ancestry_axes[0][0].set_yticks([])
ancestry_axes[0][0].imshow(img, aspect = "equal")

plotter = PlotEmbeddings(embeddings_df = true_embeddings_B,labels_df = pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
ancestry_axes[0][1].spines['left'].set_visible(False)
ancestry_axes[0][1].spines['bottom'].set_visible(False)
ancestry_axes[0][1].set_xticks([])
ancestry_axes[0][1].set_yticks([])
ancestry_axes[0][1].imshow(img, aspect = "equal")

plotter = PlotEmbeddings(embeddings_df = true_embeddings_C,labels_df = pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
ancestry_axes[0][2].spines['left'].set_visible(False)
ancestry_axes[0][2].spines['bottom'].set_visible(False)
ancestry_axes[0][2].set_xticks([])
ancestry_axes[0][2].set_yticks([])
ancestry_axes[0][2].imshow(img, aspect = "equal")

plotter = PlotEmbeddings(embeddings_df = true_embeddings_D,labels_df = pop_labels,colors_df = pop_colors,sample_id_col = "individual_id")
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = "log", xy_lims = xy_lims)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
ancestry_axes[0][3].spines['left'].set_visible(False)
ancestry_axes[0][3].spines['bottom'].set_visible(False)
ancestry_axes[0][3].set_xticks([])
ancestry_axes[0][3].set_yticks([])
ancestry_axes[0][3].imshow(img, aspect = "equal")

#%%
# Panel labels, anchored to the top-left of each labelled axis/row
row1_top = demography_ax.get_position().y1
row2_top = ancestry_axes[0][0].get_position().y1

fig.text(demography_ax.get_position().x0-0.03, row1_top, "A", fontsize=12, fontweight="bold", va="top", ha="left")
fig.text(WG_embeddings_ax.get_position().x0, row1_top, "B", fontsize=12, fontweight="bold", va="top", ha="left")
fig.text(rates_hist_ax.get_position().x0-0.03, row1_top, "C", fontsize=12, fontweight="bold", va="top", ha="left")

fig.text(ancestry_axes[0][0].get_position().x0-0.01, row2_top, "D", fontsize=12, fontweight="bold", va="top", ha="left")

ancestry_axes[0][0].set_title("A Ancestry", fontsize=10)
ancestry_axes[0][1].set_title("B Ancestry", fontsize=10)
ancestry_axes[0][2].set_title("C Ancestry", fontsize=10)
ancestry_axes[0][3].set_title("D Ancestry", fontsize=10)

filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../outputs/{filename}.pdf", dpi=500, bbox_inches="tight", pad_inches=0.01)
# %%
