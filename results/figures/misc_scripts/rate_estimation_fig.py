#%%
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

import tqdm
from results.figures.misc_scripts.plotting_style import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def data_to_fig_coords(ax, x, y):
    return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))

def inset_axes_to_fig(ax_inset, x_ax, y_ax):
    return fig.transFigure.inverted().transform(ax_inset.transAxes.transform((x_ax, y_ax)))

def init_plot(cum_rates, times, focal_pair_idx = 0, num_pairs_plotted = 1000, min_time=None, max_time=None):
    fig,ax = plt.subplots(figsize=(6,4), dpi=500)
    for i in tqdm.tqdm(range(cum_rates.shape[0])):
        ax.plot(times, cum_rates[i,:], color="grey", lw=0.1, alpha=0.1)
    ax.plot(times, cum_rates[focal_pair_idx,:], color="black", lw=1)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlabel("Time (generations ago)")
    ax.set_ylabel(r"$-\ln\left(P(T_{ij} > t)\right)$")
    ax.set_xlim(0,)
    ax.set_ylim(0,0.3)
    return fig, ax

def add_inset_with_fit(cum_rates, ax, times, delta_t, focal_time, window_size, focal_pair_idx, vertical_pos):
    t_min = min(times)
    t_max = max(times)
    window_size = int(window_size / delta_t)  # Convert window size from time to index

    x1, x2 = focal_time - window_size, focal_time + window_size
    y1, y2 = cum_rates[focal_pair_idx, (times >= x1) & (times <= x2)].min(), cum_rates[focal_pair_idx, (times >= x1) & (times <= x2)].max()

    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='black')
    ax.add_patch(rect)

    # Get axis position (in figure fraction)
    ax_pos = ax.get_position()
    ax_left = ax_pos.x0
    ax_window_size = ax_pos.width
    ax_bottom = vertical_pos  # As you want, in figure fraction
    inset_window_size = 0.1
    inset_height = 0.1

    # Compute axes fraction
    axes_fraction = (focal_time - t_min) / (t_max - t_min)
    figure_fraction = ax_left + axes_fraction * ax_window_size
    inset_left = figure_fraction - 0.5 * inset_window_size

    ax_inset = fig.add_axes([inset_left, ax_bottom, inset_window_size, inset_height])
    ax_inset.scatter(times, cum_rates[focal_pair_idx], color="black", s=1)
    ax_inset.set_xlim(x1, x2)
    ax_inset.set_ylim(y1, y2)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Box corners (main axis, data coords)
    ul_box = data_to_fig_coords(ax, x1, y2)  # upper left
    ur_box = data_to_fig_coords(ax, x2, y2)  # upper right

    # Inset corners (inset axis, axes coords)
    ll_inset = inset_axes_to_fig(ax_inset, 0, 0) # lower left
    lr_inset = inset_axes_to_fig(ax_inset, 1, 0) # lower right

    # Draw lines
    fig.add_artist(Line2D([ul_box[0], ll_inset[0]], [ul_box[1], ll_inset[1]], color='black', linewidth=1))
    fig.add_artist(Line2D([ur_box[0], lr_inset[0]], [ur_box[1], lr_inset[1]], color='black', linewidth=1))

    # Add tangent line to inset
    ax_inset = add_tangent_line(cum_rates, ax_inset, times, delta_t, focal_pair_idx, focal_time, window_size)
    
    return ax

def add_tangent_line(cum_rates, inset_ax, times, delta_t, focal_pair_idx, focal_time, window_size):
    # Define window for tangent calculation
    x_win = times[focal_time - window_size:focal_time + window_size]
    y_win = cum_rates[focal_pair_idx, focal_time - window_size:focal_time + window_size]
    
    # linear fit
    slope, intercept = np.polyfit(x_win, y_win, 1)
    
    # plot_linear_fit
    x_fit = np.array([x_win.min(), x_win.max()])
    y_fit = slope * x_fit + intercept
    inset_ax.plot(x_fit, y_fit, color="red", linestyle="--")

    mantissa, exponent = f"{slope:.2e}".split("e")
    title = (
        f"window size = {window_size*delta_t:.0f}\n"
        f"$\\lambda({focal_time}) = {mantissa}\\times10^{{{int(exponent)}}}$"
    )

    inset_ax.set_title(title, fontsize=12)
    return ax
# %%
coal_pdf_path = "../../empirical/Wohns/all.npp.2/data/coal_pdf.V1.FS1.0.maxT4000.nT4000.parquet"
coal_pdf = pd.read_parquet(coal_pdf_path)
times = coal_pdf.columns.astype(float).values
delta_t = times[1] - times[0]

S = 1 - coal_pdf.cumsum(axis=1)
cum_rates = -np.log(S).values
mean_H = cum_rates.mean(axis=0)
#%%
focal_pair_idx = 4524
num_pairs_plotted = 5000
window_size_1 = 25
focal_time_1 = 1000
window_size_2 = 100
focal_time_2 = 3000
vertical_pos = 0.75

fig,ax = init_plot(cum_rates, times, focal_pair_idx = focal_pair_idx, num_pairs_plotted = num_pairs_plotted)
ax = add_inset_with_fit(cum_rates, ax, times, delta_t, focal_time_1, window_size_1, focal_pair_idx=focal_pair_idx, vertical_pos = vertical_pos)
ax = add_inset_with_fit(cum_rates, ax, times, delta_t, focal_time_2, window_size_2, focal_pair_idx=focal_pair_idx, vertical_pos = vertical_pos)
# %%
rates_path = "../../empirical/Wohns/all.npp.2/data/rates.minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet"
rates = pd.read_parquet(rates_path).values

fig_rates, ax_rates = plt.subplots(figsize=(6,4), dpi=500)
for i in tqdm.tqdm(range(rates.shape[0])):
    ax_rates.plot(times, rates[i,:], color="grey", lw=0.1, alpha=0.1)
ax_rates.plot(times, rates[focal_pair_idx,:], color="black", lw=1)
ax_rates.set_xlabel("Time (generations ago)")
ax_rates.set_ylabel(r"Estimated rate $\hat{\lambda}_{ij}(t)$")
ax_rates.set_xlim(0,)
ax_rates.set_ylim(0,2e-4)
ax_rates.set_xticks([0, 1000, 2000, 3000, 4000])
ax_rates.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax_rates.axvline(x=focal_time_1, color="red", linestyle="--")
ax_rates.axvline(x=focal_time_2, color="red", linestyle="--")
# %%
fig.savefig("cum_rates_with_insets.png", bbox_inches="tight", dpi=500)
fig_rates.savefig("estimated_rates.png", bbox_inches="tight", dpi=500)