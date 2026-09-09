#%%
"""
Schematic describing the embeddings model:
a) coalescence rate as a function of time in the sawtooth model
b) embeddings at two timepoints
c) 3D embeddings at all timepoints
d) distances between embeddings through time

Data: results/figure_data/model_description/, written by
utils/runs/prepare_figure_data/model_description.py
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
from plotly import graph_objects as go
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from matplotlib.transforms import ScaledTranslation, Bbox
from matplotlib.patches import FancyArrowPatch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "..", "..", "src"))
os.chdir(os.path.dirname(__file__))
file_name = os.path.basename(__file__).split(".")[0]
from results.figures.plotting_style import *
from PlotEmbeddings import PlotEmbeddings

t1_color = "blue"
t2_color = "deepskyblue"
rates_color = "maroon"
distances_color = "magenta"

FIGURE = "model_description"
embeddings_df = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
distances_df = pd.read_parquet(figure_data(FIGURE, "distances.parquet"))
true_rates = pd.read_parquet(figure_data(FIGURE, "true_rates.parquet"))
mean_distances_df = pd.read_parquet(figure_data(FIGURE, "mean_distances.parquet"))
sample_colors = pd.read_csv(figure_data(FIGURE, "sample_colors.csv"))
params = pd.read_csv(figure_data(FIGURE, "params.csv")).set_index("parameter")["value"]

max_time = int(params["max_time"])
delta_t = int(params["delta_t"])
t1, t2 = int(params["t1"]), int(params["t2"])
t1_idx, t2_idx = t1 // delta_t, t2 // delta_t
# the pair the figure follows, chosen in the prep script
i, j = int(params["focal_i"]), int(params["focal_j"])
xy_lims = float(params["xy_lims"])

times = true_rates["time"].to_numpy()
rate_curve = true_rates["rate"].to_numpy()
mean_distances = mean_distances_df["mean_distance"].to_numpy()
focal_distances = mean_distances_df["focal_distance"].to_numpy()

# X[sample, component, timepoint], rebuilt from the tidy embeddings table so the
# 2D snapshot panels can index it by timepoint
X = (embeddings_df.sort_values(["individual_id", "time"])
     .set_index(["individual_id", "time"])[["X1", "X2"]]
     .to_numpy()
     .reshape(embeddings_df["individual_id"].nunique(), -1, 2)
     .transpose(0, 2, 1))

# the labels/colours the 3D panel and the 2D snapshots share
pop_labels = sample_colors[["individual_id", "role"]].rename(columns={"role": "population"})
pop_colors = pd.DataFrame({"population": ["generic", "focal"],
                           "color": ["#BEBEBE", "#000000"]})
pop_labels["color"] = pop_labels["population"].map(
    pop_colors.set_index("population")["color"])
colors = pop_labels["color"].values

# %%
#----------------------------------------------------------
# Preparing plotting parameters 
#----------------------------------------------------------
# Adding features to 3d fig
xy_lims = 140000
plotter = PlotEmbeddings(embeddings_df = embeddings_df, labels_df = pop_labels, colors_df = pop_colors, sample_id_col = "individual_id")
plotter.init_plot(xy_lims = xy_lims, min_time=10, max_time=max_time)
plotter.add_sample_trajectories(alpha = 0.6)
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
img, aspect_ratio = plotter.save_static(width = 800, height = 800)

#%%
#----------------------------------------------------------
# Constructing the figure
#----------------------------------------------------------
W = 6.5
H = W/(2*aspect_ratio)
fig = plt.figure(figsize=(W, H), dpi=500)
outer = gridspec.GridSpec(1, 4, width_ratios=[1/3, 1/3, 1, 1/3], wspace=0.2, figure=fig)
rates_ax = fig.add_subplot(outer[0])
embeddings_ax = fig.add_subplot(outer[2])
distances_ax = fig.add_subplot(outer[3])
inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.2)
t1_ax = fig.add_subplot(inner[1])
t2_ax = fig.add_subplot(inner[0])


# Plot rates over time
rates_ax.plot(rate_curve, times, color = rates_color, lw = 2)
rates_ax.set_yscale("log")
rates_ax.invert_xaxis()
rates_ax.yaxis.tick_right()
rates_ax.yaxis.set_label_position('right')
rates_ax.axhline(y=t1, color=t1_color, ls="--", lw=2)
rates_ax.axhline(y=t2, color=t2_color, ls="--", lw=2)
rates_ax.spines['left'].set_visible(False)
rates_ax.spines['right'].set_visible(True)
# Decade major ticks labelled as powers of ten, with the intra-decade subdivisions as
# unlabelled minor ticks
rates_ax.yaxis.set_major_locator(ticker.FixedLocator([1e1, 1e2, 1e3, 1e4]))
rates_ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10.0))
rates_ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
rates_ax.yaxis.set_minor_formatter(ticker.NullFormatter())
# Only two x ticks so the narrow panel stays readable
rates_ax.set_xticks([1e-5, 6e-5], labels=[r"$1\times10^{-5}$", r"$6\times10^{-5}$"])
rates_ax.tick_params(axis="both", which="major", labelsize=6)
rates_ax.tick_params(axis="y", which="minor", length=2)
# Label on the left even though the ticks are drawn on the right
rates_ax.set_ylabel("Time (generations ago)", fontsize=10)
rates_ax.yaxis.set_label_position('left')


# Plot embeddings at two time points, with the focal pair i,j joined by a segment
# coloured to match the corresponding time marker on the rates panel
for ax, t_idx, t_color in ((t1_ax, t1_idx, t1_color), (t2_ax, t2_idx, t2_color)):
    ax.scatter(X[:,0,t_idx], X[:,1,t_idx], color = colors, s = 2)
    ax.plot([X[i,0,t_idx], X[j,0,t_idx]], [X[i,1,t_idx], X[j,1,t_idx]],
            color = t_color, lw = 1, zorder = 1)
    ax.set_xlim(-xy_lims, xy_lims)
    ax.set_ylim(-xy_lims, xy_lims)
    # Axes drawn as crosshairs through the origin instead of a bounding box
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # Axis names sit at the ends of the crosshairs rather than outside the panel, so
    # they are placed in axes coordinates (the origin is at 0.5,0.5 given symmetric lims)
    label_pad = 2 / 72  # 2pt, expressed in inches for the offset transform
    ax.text(1.0, 0.5, r"$X_1$", transform=ax.transAxes + ScaledTranslation(
                0, -label_pad, fig.dpi_scale_trans),
            fontsize=6, ha="right", va="top")
    ax.text(0.5, 1.0, r"$X_2$", transform=ax.transAxes + ScaledTranslation(
                label_pad, 0, fig.dpi_scale_trans),
            fontsize=6, ha="left", va="top")


# Plot 3d embeddings
embeddings_ax.imshow(img)
embeddings_ax.axis('off')

# Plot distances over time, mirroring the rates panel but with the axis on the right
distances_ax.plot(distances_df.values.T, times, color="#D9D9D9", lw=0.4, zorder=1)
distances_ax.plot(mean_distances, times, color=distances_color, lw=2, zorder=3)
distances_ax.plot(focal_distances, times, color="black", lw=1.5, zorder=2)
# Inverse coalescent rate, i.e. N_e(t), for comparison against the recovered distances
distances_ax.plot(1 / rate_curve, times, color=rates_color, ls="--", lw=2, zorder=4)
distances_ax.set_yscale("log")
distances_ax.invert_xaxis()
distances_ax.yaxis.tick_right()
distances_ax.yaxis.set_label_position('right')
distances_ax.axhline(y=t1, color=t1_color, ls="--", lw=2)
distances_ax.axhline(y=t2, color=t2_color, ls="--", lw=2)
distances_ax.spines['left'].set_visible(False)
distances_ax.spines['right'].set_visible(True)
# Decade major ticks labelled as powers of ten, with the intra-decade subdivisions as
# unlabelled minor ticks
distances_ax.yaxis.set_major_locator(ticker.FixedLocator([1e1, 1e2, 1e3, 1e4]))
distances_ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10.0))
distances_ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
distances_ax.yaxis.set_minor_formatter(ticker.NullFormatter())
# Only two x ticks so the narrow panel stays readable
distances_ax.set_xticks([1e4, 2e5], labels=[r"$1\times10^{4}$", r"$2\times10^{5}$"])
distances_ax.tick_params(axis="both", which="major", labelsize=6)
distances_ax.tick_params(axis="y", which="minor", length=2)
distances_ax.set_ylabel("Time (generations ago)", fontsize=10)

#----------------------------------------------------------
# Arrows marking the flow from rates to snapshots to the 3D embedding
#----------------------------------------------------------
# Drawn in figure coordinates so they can span the gaps between panels; endpoints are
# read off the axes positions so they track any change to the gridspec ratios
fig.canvas.draw()
rates_box = rates_ax.get_position()
snapshots_box = Bbox.union([t1_ax.get_position(), t2_ax.get_position()])
embeddings_box = embeddings_ax.get_position()
arrow_style = dict(arrowstyle="-|>", lw=1, color="black", mutation_scale=7,
                   shrinkA=0, shrinkB=0, transform=fig.transFigure)

# Both arrows share a length and vertical position, and are anchored by their heads so
# the overhang into the following panel is consistent
arrow_y = snapshots_box.y0 + snapshots_box.height / 2
arrow_length = 0.06

# Rates panel to the midline between the two snapshot panels
arrow_1_head = snapshots_box.x0 + 0.035
fig.add_artist(FancyArrowPatch(
    (arrow_1_head - arrow_length, arrow_y), (arrow_1_head, arrow_y), **arrow_style))

# Snapshot panels to the left edge of the 3D embedding
arrow_2_head = embeddings_box.x0 - 0.005
fig.add_artist(FancyArrowPatch(
    (arrow_2_head - arrow_length, arrow_y), (arrow_2_head, arrow_y), **arrow_style))

# Legend and labels
# Panel letters sit just above the top-left corner of each panel; t2_ax carries the
# label for the stacked snapshot column so only its top panel is marked
panel_letters = ((rates_ax, "A"), (t2_ax, "B"), (embeddings_ax, "C"), (distances_ax, "D"))
# The panels have different top edges (t2_ax is half-height, embeddings_ax is an image
# axis), so the letters share one figure-space y taken from the tallest panel
letter_y = max(ax.get_position().y1 for ax, _ in panel_letters) - 0.02
for ax, letter in panel_letters:
    fig.text(ax.get_position().x0 - 0.012, letter_y, letter,
             fontsize=12, fontweight="bold", ha="left", va="bottom")

# Shared legend below the panels; proxy artists are used since the entries describe
# element types rather than any single plotted series
legend_handles = [
    Line2D([], [], color=rates_color, lw=2),
    Line2D([], [], color="black", ls="none", marker="o", ms=2),
    Line2D([], [], color=t1_color, ls="--", lw=2),
    Line2D([], [], color=t2_color, ls="--", lw=2),
    Line2D([], [], color=rates_color, ls="--", lw=2),
    Line2D([], [], color=distances_color, lw=2),
]
legend_labels = [
    r"$\lambda_{ij}(t)$",
    "Representative samples",
    rf"$t_1={t1}$ generations ago",
    rf"$t_2={t2}$ generations ago",
    r"$N_e(t)$",
    "Average distances",
]
fig.legend(legend_handles, legend_labels, ncol=3, loc="upper center",
           bbox_to_anchor=(0.55, 0.06), fontsize=10, frameon=False,
           handlelength=1.5, columnspacing=1.5, handletextpad=0.5)

# %%
fig.savefig(f"../sim_figs/{file_name}.pdf", bbox_inches="tight", dpi=500)
# %%
