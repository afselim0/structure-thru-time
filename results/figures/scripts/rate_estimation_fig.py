#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from plotting_style import *

def init_plot(coal_pdf_path):
    coal_pdf = pd.read_parquet(coal_pdf_path)
    coal_pdf = coal_pdf.iloc[:,:500]
    survival_df = 1 - coal_pdf.cumsum(axis=1)
    H = -np.log(survival_df)
    times = survival_df.columns.astype(float).values
    mean_H = H.mean(axis=0).values
    time_step = times[1] - times[0]

    fig,ax = plt.subplots(figsize=(6,4), dpi=500)
    ax.plot(times, mean_H, label="mean hazard", color="grey")
    ax.set_xticks([])
    ax.set_xlabel("Time (generations ago)")
    ax.set_ylabel(r"$-\ln\left(P(T_{ij} > t)\right)$")

    ax.set_xlim(0,)
    ax.set_ylim(0,)
    return fig, ax, mean_H, times

def add_tangent_line(ax, times, mean_H, idx, window_size):
    # Define window for tangent calculation
    time_step = times[1] - times[0]
    window_size = int(window_size / time_step)  # Convert window size from time to index
    start = int(max(0, idx - window_size // 2))
    end = int(min(len(times), idx + window_size // 2 + 1))
    x_win = times[start:end]
    y_win = mean_H[start:end]
    
    # Fit the tangent (linear fit)
    slope, intercept = np.polyfit(x_win, y_win, 1)
    
    # Tangent line over fitting window
    t_tan = np.array([x_win[0], x_win[-1]])
    # y_tan = slope * t_tan + intercept
    y_tan = mean_H[idx] + slope * (t_tan - times[idx])  # Center tangent at (times[idx], mean_H[idx])

    # Plot tangent line (centered at t0)
    ax.plot(t_tan, y_tan, color="black", linewidth=1.2)
    
    # Mark the point of tangency
    ax.plot(times[idx], mean_H[idx], marker="o", markersize=3, color="black", markeredgewidth=1.5)
    
    # Fill area under tangent line for window
    ax.fill_between(t_tan, y_tan, color="grey", alpha=0.2)
    
    # Vertical helper line
    ax.vlines(times[idx], 0, mean_H[idx], color="black", linestyle="dashed", linewidth=0.8)
    
    # Add x-tick for the tangency point
    current_xticks = ax.get_xticks()
    new_xticks = list(current_xticks) + [times[idx]]
    ax.set_xticks(sorted(set(new_xticks)))
# %%
coal_pdf_path = "../../simulations/outputs/zigzag/sim/data/coal_pdf.V1.FS1.0.maxT40000.nT2000.parquet"
fig, ax, mean_H, times = init_plot(coal_pdf_path)
add_tangent_line(ax, times, mean_H, idx=100, window_size=250)
add_tangent_line(ax, times, mean_H, idx=400, window_size=2000)
fig.savefig("../outputs/rate_estimation.png", dpi=500)
# %%
