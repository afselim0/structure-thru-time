#%%
"""
Script for generation figure describing the embeddings model in the paper
a) Rates as a funciton of time in the sawtooth model
b) Embeddings at two timepoints
c) 3D embeddings at all time points
d) distance between embeddings at different time points
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
from PIL import Image
from plotly import graph_objects as go
from matplotlib.lines import Line2D

sys.path.append("../../../src")
sys.path.append("../../../utils")
os.chdir(os.path.dirname(__file__))
sys.path.append("../misc_scripts")
file_name = os.path.basename(__file__).split(".")[0]
from plotting_style import *
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
#----------------------------------------------------------
# Computing Embeddings through time for the sawtooth model
#----------------------------------------------------------

# Example parameters for the sawtooth model
deme_file = "../../../data/demes/main/zigzag.yaml"
max_time = 40000
delta_t = 10
n_samples_per_pop = 20
N_min = 1000
N_max = 1000000
pop_labels, pop_colors = preprocess_demography(deme_file, n_samples_per_pop=n_samples_per_pop)

# Compute true coalescent rates and expand them to match the number of samples
true_rates = compute_true_rates(deme_file, max_time, delta_t)
expanded_true_rates = expand_true_rates(true_rates, pop_labels)

# Identifying two focal samples for plotting, and changing their labels and colors to make them stand out in the plots
i=0
j=37
pop_labels.iloc[i,1] = "focal"
pop_labels.iloc[j,1] = "focal"
pop_colors.iloc[0,1] = "#BEBEBE"
pop_colors.loc[len(pop_colors)] = ["focal", "#000000"]

# Compute latent space embeddings through time and distances between them
LSE = LatentSpaceEmbeddings(expanded_true_rates, N_min = N_min, N_max = N_max)
np.random.seed(0)
X_init = np.random.rand(LSE.num_samples, LSE.n_components)
X,loss = LSE.compute_embeddings_through_time(X_init)
embeddings_df = LSE.get_embeddings_df()
distances_df = compute_distances(embeddings_df)
mean_distances = distances_df.mean(axis=0).values
# %%
#----------------------------------------------------------
# Preparing plotting parameters 
#----------------------------------------------------------
DPI = 600
PLOT_WIDTH = 6.5 # To fit A4 paper width with some margins
CENTER_PLOT_RATIO = 0.5 # The ratio of the center plot (3D embeddings) width to the total figure width
xy_lims = 200000

# Choices of colors for plotting
plotly_static_name = f"../misc_figs/{file_name}c"
t1_color = "blue"
t2_color = "deepskyblue"
rates_color = "maroon"
distances_color = "magenta"

# Time points to highlight in the plots
t1 = 60
t2 = 8000
t1_idx = t1//delta_t
t2_idx = t2//delta_t


# Adding features to 3d fig
plotter = PlotEmbeddings(embeddings_df = embeddings_df, labels_df = pop_labels, colors_df = pop_colors)
plotter.init_plot()
plotter.add_sample_trajectories(alpha = 0.8)
fig_3d = plotter.fig

fig_3d.add_trace(go.Scatter3d(
    x=[X[i,0,t2_idx], X[j,0,t2_idx]],
    y=[X[i,1,t2_idx], X[j,1,t2_idx]],
    z=[t2, t2],  
    mode='lines',
    line=dict(color=t2_color, width=5)
))

fig_3d.add_trace(go.Scatter3d(
    x=[X[i,0,t1_idx], X[j,0,t1_idx]],
    y=[X[i,1,t1_idx], X[j,1,t1_idx]],
    z=[t1, t1],  
    mode='lines',
    line=dict(color=t1_color, width=5)
))

fig_3d.update_layout(showlegend=False)

plotter.save_static(plotly_static_name, format = "png", width = 800, height = 800)
static_3d_img = Image.open(f"{plotly_static_name}.png")
static_img_w, static_img_h = static_3d_img.size
aspect_ratio = static_img_h/static_img_w

#%%
#----------------------------------------------------------
# Constructing the figure
#----------------------------------------------------------
PLOT_HEIGHT = 6.5*aspect_ratio/2
SIDE_PLOT_RATIO = 1/4 - aspect_ratio/8
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
# ax_a.plot(y,x, color = "#BEBEBE")
ax_a.plot(y,x, color = rates_color)
ax_a.invert_xaxis()  
ax_a.set_yscale('log')
ax_a.yaxis.set_ticks_position('right')
ax_a.yaxis.set_label_position('right')
ax_a.spines['left'].set_visible(False)
ax_a.spines['right'].set_visible(True)
x_ticks = [5e-5,5e-6]
x_ticks_labels = [r"$5 \times 10^{-5}$", r"$5 \times 10^{-6}$"]
y_ticks = [t1, t2]
y_labels = [r"$t_1$", r"$t_2$"]

# ax_a.set_xticks(x_ticks)                     # Remove x ticks
# ax_a.set_xticklabels(x_ticks_labels)
ax_a.set_yticks(y_ticks)                     # Remove y ticks
ax_a.set_yticklabels(y_labels)                # Remove y tick labels
ytick_labels = ax_a.get_yticklabels() 
ytick_labels[0].set_color(t1_color)
ytick_labels[1].set_color(t2_color)

# ax_a.set_yticks([])                     # Remove y ticks
# ax_a.yaxis.set_minor_locator(plt.NullLocator())
ax_a.set_ylabel('')                     # Remove x axis label (if any)
ax_a.set_xlabel(r"$\lambda_{ij}(t)$")
ax_a.axhline(x[t1_idx], color = t1_color, linestyle = "--")
ax_a.axhline(x[t2_idx], color = t2_color, linestyle = "--")
# ax_a.set_ylabel("Time (generations ago)")

# b): Embeddings at two representative time points
ax_b = fig.add_subplot(gs[0, 1])
ax_b.scatter(X[:,0,t2_idx], X[:,1,t2_idx], color = "#BEBEBE", label = "focal", s=1)
ax_b.scatter(X[i,0,t2_idx], X[i,1,t2_idx], color = "#000000", label = "focal", s=5)
ax_b.scatter(X[j,0,t2_idx], X[j,1,t2_idx], color = "#000000", label = "focal", s=5)
ax_b.set_xticks([])  
ax_b.set_yticks([])
ax_b.set_xlim(-xy_lims, xy_lims)
ax_b.set_ylim(-xy_lims, xy_lims)

ax_b.spines['left'].set_visible(True)
ax_b.spines['bottom'].set_visible(True)
ax_b.spines['left'].set_position('center')
ax_b.spines['bottom'].set_position('center')
ax_b.spines['right'].set_color('none')
ax_b.spines['top'].set_color('none')

ax_b.set_xlabel(r"$X_1$", loc = "right")
ax_b.set_ylabel(r"$X_2$" , loc = "top", rotation = 360, labelpad= -10)
ax_b.plot([X[i,0,t2_idx], X[j,0,t2_idx]], [X[i,1,t2_idx], X[j,1,t2_idx]], color = t2_color, ls = "--", lw = 1)

pos = ax_b.get_position()
new_pos = [pos.x0 + 0.012, pos.y0, pos.width, pos.height]
ax_b.set_position(new_pos)


ax_c = fig.add_subplot(gs[1, 1])
ax_c.scatter(X[:,0,t1_idx], X[:,1,t1_idx], color = "#BEBEBE", label = "focal", s=1)
ax_c.scatter(X[i,0,t1_idx], X[i,1,t1_idx], color = "#000000", label = "focal", s=5)
ax_c.scatter(X[j,0,t1_idx], X[j,1,t1_idx], color = "#000000", label = "focal", s=5)
ax_c.set_xticks([])  
ax_c.set_yticks([])

ax_c.spines['left'].set_visible(True)
ax_c.spines['bottom'].set_visible(True)
ax_c.spines['left'].set_position('center')
ax_c.spines['bottom'].set_position('center')
ax_c.spines['right'].set_color('none')
ax_c.spines['top'].set_color('none')

ax_c.set_xlim(-xy_lims, xy_lims)
ax_c.set_ylim(-xy_lims, xy_lims)
ax_c.set_xlabel(r"$X_1$", loc = "right")
ax_c.set_ylabel(r"$X_2$", loc = "top", rotation = 360, labelpad= -10)

ax_c.plot([X[i,0,t1_idx], X[j,0,t1_idx]], [X[i,1,t1_idx], X[j,1,t1_idx]], color = t1_color, ls = "--", lw = 1)
pos = ax_c.get_position()
new_pos = [pos.x0 + 0.012, pos.y0, pos.width, pos.height]
ax_c.set_position(new_pos)



# c) 3D embeddings at all time points
plotter.save_static(plotly_static_name, format = "png", width = 0.5*PLOT_HEIGHT*DPI, height = 0.5*PLOT_HEIGHT*DPI)
ax_d = fig.add_subplot(gs[:, 2])
ax_d.axis("off")
# for spine in ax_d.spines.values():
#     spine.set_visible(False)
img = Image.open(plotly_static_name + ".png")
ax_d.imshow(img, aspect = "equal")

# pos = ax_d.get_position()
# print("Original position:", pos.bounds)
# removed_pad = 0.025
# ax_d.set_position([pos.x0-0.025, pos.y0-0.1, pos.width*1.2, pos.height*1.2])  # tweak as needed, e.g. add/subtract a small margin



# d) distance between embeddings at through time
x = true_rates.columns.values
y1 = mean_distances
y2 = 1/(true_rates.values[0])
ax_e = fig.add_subplot(gs[:, 3])
ax_e.plot(y1,x, color = distances_color, label = "average distances")
ax_e.plot(y2,x, color = rates_color, ls = "--", label = r"$N_e(t)$")
for row in distances_df.itertuples():
    ax_e.plot(row[1:], x, color = "grey", lw=0.01)
ax_e.plot(distances_df.loc[(i,j)], x, color = "black", lw=1, label = "focal pair")

ax_e.invert_xaxis()  
ax_e.set_yscale('log')
ax_e.yaxis.set_ticks_position('right')
ax_e.yaxis.set_label_position('right')
ax_e.spines['left'].set_visible(False)
ax_e.spines['right'].set_visible(True)
ax_e.tick_params(axis='y')  # Choose your desired font size
x_ticks = [2e4,2e5]
x_ticks_labels = [r"$2 \times 10^{4}$", r"$2 \times 10^{5}$"]
# ax_e.set_xticks(x_ticks)                     # Remove x ticks
# ax_e.set_xticklabels(x_ticks_labels)
ax_e.set_xlabel(r"$\Vert X_i(t) - X_j(t) \Vert$")
ax_e.axhline(x[t1_idx], color = t1_color, linestyle = "--")
ax_e.axhline(x[t2_idx], color = t2_color, linestyle = "--")
ax_e.set_ylabel("Time (generations ago)", rotation=270, labelpad=15)
# ax_e.legend(fontsize=8, loc = [0,-0.4], frameon=False)


# Adding panel labels
x_starts = [ax.get_position().x0 for ax in fig.axes]
x_centers = []
for k, ax in enumerate(fig.axes):
    bbox = ax.get_position()
    x_center = (bbox.x0 + bbox.x1) / 2
    x_centers.append(x_center)

fig.text(x_starts[0] + 0.01, 0.95, "a)", fontweight='black', va='top', ha='right')
fig.text(x_starts[1] + 0.01, 0.95, "b)", fontweight='black', va='top', ha='right')
fig.text(x_starts[3] + 0.01, 0.95, "c)", fontweight='black', va='top', ha='right')
fig.text(x_starts[4] + 0.01, 0.95, "d)", fontweight='black', va='top', ha='right')

# Add arrows between panels
ax_a.annotate(
    '',
    xy=(x_starts[0]+0.04, 0.75), xytext=(x_starts[0]+0.004, 0.75),
    xycoords='figure fraction',
    arrowprops=dict(arrowstyle='->', linewidth=2),
    zorder=10
)

ax_a.annotate(
    '',
    xy=(x_starts[1]+0.04, 0.75), xytext=(x_starts[1]+0.004, 0.75),
    xycoords='figure fraction',
    arrowprops=dict(arrowstyle='->', linewidth=2),
    zorder=10
)

# Updating whole figure legend
rate_handle = Line2D([], [], color=rates_color, linestyle='-')
rate_label = r'$\lambda_{ij}(t)$'
t1_handle = Line2D([], [], color=t1_color, linestyle='--', marker='o', markersize=5)
t1_label = rf'$t_1$={t1} generations'
t2_handle = Line2D([], [], color=t2_color, linestyle='--', marker='o', markersize=5)
t2_label = rf'$t_2$={t2} generations'
focal_pair_handle = Line2D([], [], color='black', linestyle='-', marker='o', markersize=5)
focal_pair_label = 'focal pair'
Ne_handle = Line2D([], [], color=rates_color, linestyle='--')
Ne_label = r'$N_e(t)$'
average_d_handle = Line2D([], [], color=distances_color, linestyle='-')
average_d_label = 'average distances'

handles = [rate_handle,focal_pair_handle, t1_handle, t2_handle, Ne_handle, average_d_handle]
labels  = [rate_label, focal_pair_label, t1_label, t2_label, Ne_label, average_d_label]
fig.legend(handles, labels,loc='lower center', bbox_to_anchor=(0.55, -0.2), frameon=False, ncol=3)

fig.savefig(f"../outputs/{file_name}.png", dpi=DPI)
# %%
