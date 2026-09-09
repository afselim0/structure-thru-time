#%%
"""
The complex four-population example: demography, embeddings through time, how well
the reconstructed rates match the estimated ones, and how the fit varies with p.

Data: results/figure_data/complex_4_pops/, written by
utils/runs/prepare_figure_data/complex_4_pops.py
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from results.figures.plotting_style import *
import sys
import demes
import demesdraw
import os
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils/snakemake_utils")
from PlotEmbeddings import *
import matplotlib.ticker as ticker
from evaluate_performance import reconstructed_vs_input_hist2d

FIGURE = "complex_4_pops"
demes_file_path = "../../../data/demes/complex_4_pops.yaml"

pop_colors = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))
pop_labels = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
estimated_rates = pd.read_parquet(figure_data(FIGURE, "estimated_rates.parquet"))
reconstructed_rates = pd.read_parquet(figure_data(FIGURE, "reconstructed_rates.parquet"))
scan_p = pd.read_csv(figure_data(FIGURE, "scan_p.csv"))
plot_params = pd.read_csv(figure_data(FIGURE, "plot_params.csv")).set_index("parameter")["value"]

scan_p_input = scan_p[scan_p["vs"] == "input"]
scan_p_true = scan_p[scan_p["vs"] == "true"]
min_time = float(plot_params["min_time"])
max_time = float(plot_params["max_time"])
time_scale = plot_params["time_scale"]
N_max = float(plot_params["N_max"])

# %%
#----------------------------------------------------------
# Constructing the figure
#----------------------------------------------------------
plotter = PlotEmbeddings(
    embeddings_df = embeddings,
    labels_df = pop_labels,
    colors_df = pop_colors,
    sample_id_col = "individual_id"
)
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = time_scale)
plotter.add_sample_trajectories()
# the panel is 3in tall and rendered at 500 dpi; the cropped aspect ratio depends on
# this canvas size, so the probe and the final render below must share it
RENDER_KW = dict(width=1500, height=1500)
_, aspect_ratio = plotter.save_static(**RENDER_KW, scale=1)  # cheap probe: scale does not affect the ratio


filename = os.path.basename(__file__).split(".")[0]
fig_w, fig_h = 6, 6
row1_height_in = 3  # fixed height for row 1; row 2 is whatever remains

# embeddings_ax (row 1, col 1) width follows the rendered image's aspect ratio
# at the fixed row height; demography_ax (row 1, col 0) takes the leftover width
embeddings_width_in = row1_height_in * aspect_ratio
demography_width_in = fig_w - embeddings_width_in

fig = plt.figure(figsize=(fig_w, fig_h), dpi=500)
# outer grid only fixes row heights; each row gets its own column widths below
outer = gridspec.GridSpec(
    nrows=2, ncols=1,
    height_ratios=[row1_height_in, fig_h - row1_height_in],
    hspace=0,
    left=0, right=1, top=1, bottom=0,  # span the full figure; height_ratios (in) map to true inches
    figure=fig,
)
row1 = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=2, subplot_spec=outer[0],
    width_ratios=[demography_width_in, embeddings_width_in], wspace=0.2,
)
row2 = gridspec.GridSpecFromSubplotSpec(
    nrows=1, ncols=2, subplot_spec=outer[1],
    width_ratios=[1, 1], wspace=0,
)
demography_ax = fig.add_subplot(row1[0, 0])
embeddings_ax = fig.add_subplot(row1[0, 1])
input_vs_reconstructed_ax = fig.add_subplot(row2[0, 0])
varying_p_ax = fig.add_subplot(row2[0, 1])
# a) demography plot
graph = demes.load(demes_file_path)
pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
positions = {"A": 3e5, "B": 2e5, "C": 1e5, "D": 0,
             "AB": 2.5e5, "ABC": 1.5e5, "ABCD": 5e4}
demesdraw.tubes(
    graph,
    ax=demography_ax,
    colours=pop_colors_dict,
    max_time=max_time,
    log_time=time_scale,
    positions=positions,
    labels = "xticks"
)
demography_ax.set_ylim(bottom=min_time)
demography_ax.set_ylabel("Time (generations ago)", fontsize=11)
demography_ax.text(-0.1, 1, "A", transform=demography_ax.transAxes,
           fontsize=12, fontweight="bold", va="top", ha="right")

# b) CoaSTT embeddings
plotter = PlotEmbeddings(
    embeddings_df = embeddings,
    labels_df = pop_labels,
    colors_df = pop_colors,
    sample_id_col = "individual_id"
)
plotter.init_plot(min_time = min_time, max_time = max_time, time_scale = time_scale)
plotter.add_sample_trajectories()
img, _ = plotter.save_static(**RENDER_KW)
img.save("../sim_figs_png/complex_4_pops_embeddings.png")
embeddings_ax.set_axis_off()
embeddings_ax.imshow(img, aspect="equal")
embeddings_ax.text(0, 1, "B", transform=embeddings_ax.transAxes,
           fontsize=12, fontweight="bold", va="top", ha="right")

# c) estimated vs reconstructed rates
reconstructed_vs_input_hist2d(estimated_rates, reconstructed_rates, N_max, ax=input_vs_reconstructed_ax,
                               x_label="Estimated Rates", draw_colorbar=True, n_bins=200)
input_vs_reconstructed_ax.text(-0.1, 1, "C", transform=input_vs_reconstructed_ax.transAxes,
           fontsize=12, fontweight="bold", va="top", ha="right")

# d) varying p
varying_p_ax_twin = varying_p_ax.twinx()
varying_p_ax.plot(scan_p_input['p'], -scan_p_input['loss'], color='black', label='Log Likelihood')
varying_p_ax_twin.plot(scan_p_input['p'], scan_p_true['r2'], color='red', label=rf'$r^2$')
varying_p_ax_twin.plot(scan_p_input['p'], scan_p_true['r2_log'], color='blue', label=rf'$r^2$(log scale)')
varying_p_ax_twin.plot(scan_p_input['p'], scan_p_true['spearman_rho'], color='pink', label=rf'Spearman $\rho$')
varying_p_ax.set_xlabel("p", fontsize=11)
varying_p_ax.set_ylabel("Log Likelihood", fontsize=11)
varying_p_ax_twin.set_ylabel("Correlation", fontsize=11)
# twinx() duplicates the shared x-axis tick labels on both axes; keep only one copy
varying_p_ax_twin.tick_params(axis='x', labelbottom=False)
# rcParams disables the right spine globally, but the twin needs it to anchor its ticks
varying_p_ax_twin.spines['right'].set_visible(True)
handles = varying_p_ax.get_legend_handles_labels()[0] + varying_p_ax_twin.get_legend_handles_labels()[0]
labels = varying_p_ax.get_legend_handles_labels()[1] + varying_p_ax_twin.get_legend_handles_labels()[1]
# mark p=1 on each line with a point in its own color, instead of a vertical guide line
p1_input_idx = (scan_p_input['p'] - 1).abs().idxmin()
p1_true_idx = (scan_p_true['p'] - 1).abs().idxmin()
varying_p_ax.plot(scan_p_input['p'][p1_input_idx], -scan_p_input['loss'][p1_input_idx],
                   'o', color='black', markersize=5, zorder=5)
varying_p_ax_twin.plot(scan_p_true['p'][p1_true_idx], scan_p_true['r2'][p1_true_idx],
                        'o', color='red', markersize=5, zorder=5)
varying_p_ax_twin.plot(scan_p_true['p'][p1_true_idx], scan_p_true['r2_log'][p1_true_idx],
                        'o', color='blue', markersize=5, zorder=5)
varying_p_ax_twin.plot(scan_p_true['p'][p1_true_idx], scan_p_true['spearman_rho'][p1_true_idx],
                        'o', color='pink', markersize=5, zorder=5)
varying_p_ax.text(-0.25, 1, "D", transform=varying_p_ax.transAxes,
           fontsize=12, fontweight="bold", va="top", ha="right")
varying_p_ax.set_xticks([0,1,2,3,4])
varying_p_ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
varying_p_ax_twin.set_ylim(0,1)

varying_p_ax_twin.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.16, 0), fontsize=8, frameon=False)
# %%
fig.savefig(f"../sim_figs/{filename}.pdf", bbox_inches="tight")
# %%
