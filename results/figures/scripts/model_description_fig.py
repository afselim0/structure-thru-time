#%%
"""
Script for generation figure describing the embeddings model in the paper
a) Rates as a funciton of time in the sawtooth model
b) Embeddings at two timepoints
c) 3D embeddings at all time points
d) distance between embeddings at different time points
"""
# ----------Fixed Params----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import msprime
import os
import sys
from PIL import Image

from plotting_style import *
sys.path.append("../../../src")
sys.path.append("../../../utils")
from compute_rates_true import compute_true_rates, expand_true_rates
from preprocess_demography import preprocess_demography
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
from PlotEmbeddings import PlotEmbeddings

def compute_distances(embeddings_df):
    embeddings_pairs = embeddings_df.merge(embeddings_df, on="time", suffixes=("_1", "_2"))
    embeddings_pairs = embeddings_pairs[embeddings_pairs["sample_id_1"] < embeddings_pairs["sample_id_2"]]
    embeddings_pairs["distance"] = np.sqrt(
    (embeddings_pairs["X1_1"] - embeddings_pairs["X1_2"])**2 +
    (embeddings_pairs["X2_1"] - embeddings_pairs["X2_2"])**2)

    D_df = (embeddings_pairs
    .set_index(["sample_id_1", "sample_id_2", "time"])["distance"]
    .unstack("time"))

    return D_df
# %%
deme_file = "../../../data/demes/main/zigzag.yaml"
max_time = 40000
delta_t = 10
n_samples_per_pop = 20
pop_labels, pop_colors = preprocess_demography(deme_file, n_samples_per_pop=n_samples_per_pop)
#%%
true_rates = compute_true_rates(deme_file, max_time, delta_t)
expanded_true_rates = expand_true_rates(true_rates, pop_labels)
# %%
N_min = 1000
N_max = 1000000
LSE = LatentSpaceEmbeddings(expanded_true_rates, N_min = N_min, N_max = N_max)
np.random.seed(0)
X_init = np.random.rand(LSE.num_samples, LSE.n_components)
X,loss = LSE.compute_embeddings_through_time(X_init)
embeddings_df = LSE.get_embeddings_df()
distances_df = compute_distances(embeddings_df)
mean_distances = distances_df.mean(axis=0).values
#%%
plotly_static_name = "model_description_3d"
i = 0
j = 37

pop_labels.iloc[i,1] = "focal"
pop_labels.iloc[j,1] = "focal"
pop_colors.iloc[0,1] = "#BEBEBE"
pop_colors.loc[len(pop_colors)] = ["focal", "#000000"]
plotter = PlotEmbeddings(embeddings_df = embeddings_df, labels_df = pop_labels, colors_df = pop_colors)
plotter.init_plot()
plotter.add_sample_trajectories(alpha = 0.8)
# plotter.fig.show()
plotter.save_static(plotly_static_name, format = "png", width = 800, height = 800)
static_3d_img = Image.open(f"{plotly_static_name}.png")
static_img_w, static_img_h = static_3d_img.size
aspect_ratio = static_img_h/static_img_w
# %%
# Figure!
t1= 130//delta_t
t2 = 8530//delta_t
xy_lims = 200000

PLOT_WIDTH = 6.5
PLOT_HEIGHT = 6.5*aspect_ratio/2
CENTER_PLOT_RATIO = 0.5
SIDE_PLOT_RATIO = 1/4 - aspect_ratio/8
DPI = 600

fig = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)


gs = gridspec.GridSpec(
    nrows=2, ncols=4,
    width_ratios=[SIDE_PLOT_RATIO, aspect_ratio/4, CENTER_PLOT_RATIO, SIDE_PLOT_RATIO],
    height_ratios=[aspect_ratio/4, aspect_ratio/4],
    wspace = 0.1, hspace = 0.2
)

# a): sawtooth coalescent rates as a function of time
x = true_rates.columns.values
y = true_rates.values[0]
ax_a = fig.add_subplot(gs[:, 0])
ax_a.plot(y,x, color = "#BEBEBE")
ax_a.invert_xaxis()  
ax_a.set_yscale('log')
ax_a.yaxis.set_ticks_position('right')
ax_a.yaxis.set_label_position('right')
ax_a.spines['left'].set_visible(False)
ax_a.spines['right'].set_visible(True)
ax_a.set_xticks([])                     # Remove x ticks
ax_a.set_yticklabels([])                     # Remove y tick labels
# ax_a.set_yticks([])                     # Remove y ticks
# ax_a.yaxis.set_minor_locator(plt.NullLocator())
ax_a.set_ylabel('')                     # Remove x axis label (if any)
ax_a.set_xlabel(r"$\lambda_{ij}(t)$", size=10)
ax_a.axhline(x[t1], color = "#000000", linestyle = "--")
ax_a.axhline(x[t2], color = "#000000", linestyle = "--")

# b): Embeddings at two representative time points
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(X[:,0,t2], X[:,1,t2], color = "#BEBEBE", label = "focal", s=5)
ax_b.scatter(X[i,0,t2], X[i,1,t2], color = "#000000", label = "focal", s=5)
ax_b.scatter(X[j,0,t2], X[j,1,t2], color = "#000000", label = "focal", s=5)
ax_b.set_xticks([])  
ax_b.set_yticks([])
ax_b.set_xlim(-xy_lims, xy_lims)
ax_b.set_ylim(-xy_lims, xy_lims)

ax_b.spines['left'].set_position('center')
ax_b.spines['bottom'].set_position('center')
ax_b.spines['right'].set_color('none')
ax_b.spines['top'].set_color('none')

ax_b.set_xlabel(r"$X_1$", size=10, loc = "right")
ax_b.set_ylabel(r"$X_2$", size=10 , loc = "top")


ax_c = fig.add_subplot(gs[1, 1])
ax_c.scatter(X[:,0,t1], X[:,1,t1], color = "#BEBEBE", label = "focal", s=5)
ax_c.scatter(X[i,0,t1], X[i,1,t1], color = "#000000", label = "focal", s=5)
ax_c.scatter(X[j,0,t1], X[j,1,t1], color = "#000000", label = "focal", s=5)
ax_c.set_xticks([])  
ax_c.set_yticks([])

ax_c.spines['left'].set_position('center')
ax_c.spines['bottom'].set_position('center')
ax_c.spines['right'].set_color('none')
ax_c.spines['top'].set_color('none')

ax_c.set_xlim(-xy_lims, xy_lims)
ax_c.set_ylim(-xy_lims, xy_lims)
ax_c.set_xlabel(r"$X_1$", size=10, loc = "right")
ax_c.set_ylabel(r"$X_2$", size=10 , loc = "top")


# c) 3D embeddings at all time points
plotter.save_static("fig1c", format = "png", width = 0.5*PLOT_HEIGHT*DPI, height = 0.5*PLOT_HEIGHT*DPI)
ax_d = fig.add_subplot(gs[:, 2])
ax_d.axis("off")
# for spine in ax_d.spines.values():
#     spine.set_visible(False)
img = Image.open("fig1c.png")
ax_d.imshow(img, aspect = "equal")

# pos = ax_d.get_position()
# print("Original position:", pos.bounds)
# removed_pad = 0.025
# ax_d.set_position([pos.x0-0.025, pos.y0-0.1, pos.width*1.2, pos.height*1.2])  # tweak as needed, e.g. add/subtract a small margin



# d) distance between embeddings at through time
x = true_rates.columns.values
y = mean_distances
ax_e = fig.add_subplot(gs[:, 3])
ax_e.plot(y,x, color = "#BEBEBE")
ax_e.invert_xaxis()  
ax_e.set_yscale('log')
ax_e.yaxis.set_ticks_position('right')
ax_e.yaxis.set_label_position('right')
ax_e.spines['left'].set_visible(False)
ax_e.spines['right'].set_visible(True)
ax_e.tick_params(axis='y', labelsize=10)  # Choose your desired font size
ax_e.set_xticks([])                     # Remove x ticks
ax_e.set_xlabel(r"$\bar{d}_{ij}(t)$", size=10)
ax_e.axhline(x[t1], color = "#000000", linestyle = "--")
ax_e.axhline(x[t2], color = "#000000", linestyle = "--")
ax_e.set_ylabel("Time (generations ago)", size = 10)

fig.text(0.1, 0.85, 'a)', fontsize=10, fontweight='black')
fig.text(0.25, 0.85, 'b)', fontsize=10, fontweight='black')
fig.text(0.25, 0.45, 'c)', fontsize=10, fontweight='black')
fig.text(0.4, 0.85, 'd)', fontsize=10, fontweight='black')
fig.text(0.8, 0.85, 'e)', fontsize=10, fontweight='black')

plt.tight_layout()
plt.show()

fig.savefig("../outputs/model_description_figure.png", dpi=DPI)
# %%
