#%%
#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import sys
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import spearmanr
from plot_rates import plot_rates_multi_figs, plot_rates_single_fig

sys.path.append("../../src")
sys.path.append("../src")
from optimization_functions import compute_f
from plotting_style import *
import argparse
import matplotlib.pyplot as plt
import os

def reconstruct_rates(embeddings, N_min, N_max):
    embeddings_pairs = embeddings.merge(embeddings, on="time", suffixes=("_1", "_2"))
    embeddings_pairs = embeddings_pairs[embeddings_pairs["sample_id_1"] < embeddings_pairs["sample_id_2"]]
    embeddings_pairs["distance"] = np.sqrt(
    (embeddings_pairs["X1_1"] - embeddings_pairs["X1_2"])**2 +
    (embeddings_pairs["X2_1"] - embeddings_pairs["X2_2"])**2)

    D_df = (embeddings_pairs
    .set_index(["sample_id_1", "sample_id_2", "time"])["distance"]
    .unstack("time"))

    reconstructed_rates,_ = compute_f(D_df, N_min)
    return reconstructed_rates, D_df

def reconstructed_vs_input_all_times(input_rates, reconstructed_rates, N_max):
    input_rates += 1/N_max
    cmap = cm.viridis
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    assert input_rates.shape == reconstructed_rates.shape, "Input and reconstructed rates must have the same shape"
    assert np.all(input_rates.columns.values == reconstructed_rates.columns.values), "Input and reconstructed rates must have the same time points"
    assert input_rates.index.equals(reconstructed_rates.index), "Input and reconstructed rates must have the same samples"
    times = input_rates.columns.astype(float)

    fig, ax = plt.subplots(figsize=(6,4))

    # Normalize times to [0, 1] for the colormap
    norm = colors.Normalize(vmin=min(times), vmax=max(times))
    for t, time in enumerate(times):
        ax.scatter(
            input_rates.iloc[:, t],
            reconstructed_rates.iloc[:, t],
            alpha=0.4,
            s=1,
            color=cmap(norm(time))
        )
    ax.set_xlabel("input rates + 1/N_max")
    ax.set_ylabel("reconstructed rates")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    # y=x line for reference
    ax.plot([np.min(input_rates.values), np.max(input_rates.values)], [np.min(input_rates.values), np.max(input_rates.values)], 'k--', lw=2)

    # Colorbar lenged
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for older matplotlib
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("time")

    ## Add r^2 value to the plot
    input_rate_values = input_rates.values.flatten()
    reconstructed_rate_values = reconstructed_rates.values.flatten()
    valid = np.isfinite(input_rate_values) & np.isfinite(reconstructed_rate_values) & (input_rate_values > 0) & (reconstructed_rate_values > 0)
    r_squared     = np.corrcoef(input_rate_values[valid], reconstructed_rate_values[valid])[0, 1] ** 2
    log_r_squared = np.corrcoef(np.log(input_rate_values[valid]), np.log(reconstructed_rate_values[valid]))[0, 1] ** 2
    corr, p_value = spearmanr(input_rate_values[valid], reconstructed_rate_values[valid])
    ax.text(0.05, 0.95, f"$r^2$ = {r_squared:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, f"$r^2$ (log) = {log_r_squared:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.85, f"Spearman $\\rho$ = {corr:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

    return fig

def reconstructed_vs_input_query(sample_id, input_rates, reconstructed_rates, pop_labels, pop_colors, N_max):
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    assert input_rates.shape == reconstructed_rates.shape, "Input and reconstructed rates must have the same shape"
    assert np.all(input_rates.columns.values == reconstructed_rates.columns.values), "Input and reconstructed rates must have the same time points"
    assert input_rates.index.equals(reconstructed_rates.index), "Input and reconstructed rates must have the same samples"

    pairs = input_rates.index.values
    pair_mask = np.array([(sample_id in pair) for pair in pairs])
    pair_indices = np.where(pair_mask)[0]
    other_sample_ids = np.array([pair[0] if pair[1] == sample_id else pair[1] for pair in pairs[pair_mask]])
    other_sample_labels = pop_labels.set_index("sample_id").loc[other_sample_ids]["population"].values
    other_sample_colors = pop_colors.set_index("population").loc[other_sample_labels]["color"].values

    filtered_input_rates = input_rates.iloc[pair_indices].values
    filtered_reconstructed_rates = reconstructed_rates.iloc[pair_indices].values

    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(filtered_input_rates.shape[0]):
        ax.scatter(
            filtered_input_rates[i, :],
            filtered_reconstructed_rates[i, :],
            alpha=0.4,
            s=1,
            color=other_sample_colors[i]
        )
    ax.set_xlabel("input rates + 1/N_max")
    ax.set_ylabel("reconstructed rates")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    # y=x line for reference
    ax.plot([np.min(filtered_input_rates), np.max(filtered_input_rates)], [np.min(filtered_input_rates), np.max(filtered_input_rates)], 'k--', lw=2)
    ## Add r^2 value to the plot
    input_rate_values = filtered_input_rates.flatten()
    reconstructed_rate_values = filtered_reconstructed_rates.flatten()
    r_squared = np.corrcoef(input_rate_values, reconstructed_rate_values)[0, 1] ** 2
    corr, p_value = spearmanr(input_rate_values, reconstructed_rate_values)
    ax.text(0.05, 0.95, f"$r^2$ = {r_squared:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, f"Spearman $\\rho$ = {corr:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

    return fig

def reconstructed_vs_input_interactive(input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, top_k=2000, min_pairs_per_slice=10):
    input_rates = input_rates.copy()
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates = reconstructed_rates.copy()
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    pairs = input_rates.index
    times = input_rates.columns.values

    log_ratio = np.abs(np.log(reconstructed_rates.values / input_rates.values))

    # Z-score within each time slice; skip slices with too few valid pairs
    z_scores = np.full_like(log_ratio, np.nan)
    for t_idx in range(len(times)):
        col = log_ratio[:, t_idx]
        valid = np.isfinite(col)
        if valid.sum() < min_pairs_per_slice:
            continue
        std_t = col[valid].std()
        if std_t < 1e-12:
            continue
        z_scores[valid, t_idx] = (col[valid] - col[valid].mean()) / std_t

    # Select top K by z-score
    flat_z = z_scores.copy()
    flat_z[~np.isfinite(flat_z)] = -np.inf
    flat_indices = np.argsort(flat_z, axis=None)[-top_k:][::-1]
    pair_indices, time_indices = np.unravel_index(flat_indices, z_scores.shape)

    # Capture full-data axis limits before subsetting (matches static plot autoscale)
    all_vals = np.concatenate([input_rates.values.ravel(), reconstructed_rates.values.ravel()])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    rate_min = float(all_vals.min())
    rate_max = float(all_vals.max())

    pop_map = pop_labels.set_index("sample_id")["population"]
    df = pd.DataFrame({
        "sample_1": [pairs[i][0] for i in pair_indices],
        "sample_2": [pairs[i][1] for i in pair_indices],
        "time":               times[time_indices],
        "input_rate":         input_rates.values[pair_indices, time_indices],
        "reconstructed_rate": reconstructed_rates.values[pair_indices, time_indices],
        "z_score":            z_scores[pair_indices, time_indices],
    })
    df["pop_1"] = df["sample_1"].map(pop_map).fillna("unknown")
    df["pop_2"] = df["sample_2"].map(pop_map).fillna("unknown")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[rate_min, rate_max], y=[rate_min, rate_max],
        mode="lines", line=dict(color="black", dash="dash", width=2),
        name="y = x", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=df["input_rate"],
        y=df["reconstructed_rate"],
        mode="markers",
        marker=dict(
            size=5, color=df["time"], colorscale="Viridis",
            showscale=True, colorbar=dict(title="Time"), opacity=0.7,
        ),
        customdata=np.stack([
            df["sample_1"].astype(str), df["sample_2"].astype(str),
            df["pop_1"].astype(str),    df["pop_2"].astype(str),
            df["time"].values,          df["z_score"].values,
        ], axis=1),
        hovertemplate=(
            "<b>%{customdata[2]} — %{customdata[3]}</b><br>"
            "Samples: %{customdata[0]} / %{customdata[1]}<br>"
            "Time: %{customdata[4]:.1f}<br>"
            "Input rate: %{x:.3e}<br>"
            "Reconstructed rate: %{y:.3e}<br>"
            "Z-score: %{customdata[5]:.2f}"
            "<extra></extra>"
        ),
        name=f"Top-{top_k} outliers",
    ))
    fig.update_layout(
        xaxis=dict(title="Input rate + 1/N_max", type="log", range=[math.log10(rate_min), math.log10(rate_max)]),
        yaxis=dict(title="Reconstructed rate",   type="log", range=[math.log10(rate_min), math.log10(rate_max)]),
        title=f"Top-{top_k} outlier (pair × time) points by within-timepoint z-score",
    )
    return fig


def find_percent_error(input_rates, reconstructed_rates, N_max):
    input_rates += 1/N_max
    error = np.abs(reconstructed_rates - input_rates) / np.abs(input_rates)

    return error
    
def main():
    argparser = argparse.ArgumentParser(description="Evaluate performance of rate reconstruction from embeddings")
    argparser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings parquet file")
    argparser.add_argument("--rates", type=str, required=True, help="Path to true rates parquet file")
    argparser.add_argument("--colors", type=str, required=True, help="Path to population colors csv file")
    argparser.add_argument("--labels", type=str, required=True, help="Path to population labels csv file")
    argparser.add_argument("--N_min", type=float, default=None, help="N_min used to compute embeddings")
    argparser.add_argument("--N_max", type=float, default=None, help="N_max used to compute embeddings")
    argparser.add_argument("--data_output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--plot_output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--data_output_name", type=str, required=True, help="Name of the output file")
    argparser.add_argument("--plot_output_name", type=str, required=True, help="Name of the output file")
    argparser.add_argument("--min_time", type=float, default=0, help="Minimum time for plotting")
    argparser.add_argument("--max_rate_plot", type=float, default=None, help="Maximum rate for plotting")
    argparser.add_argument("--time_scale", default="linear", help="Time scale for demography plot, linear or log", type=str)
    args = argparser.parse_args()

    # Loading data
    N_min = args.N_min
    N_max = args.N_max
    embeddings = pd.read_parquet(args.embeddings)
    rates = pd.read_parquet(args.rates)
    rates.columns = rates.columns.astype(float)
    pop_colors = pd.read_csv(args.colors)
    pop_labels = pd.read_csv(args.labels)
    reconstructed_rates,D_df = reconstruct_rates(embeddings, N_min=N_min, N_max=N_max)
    reconstructed_rates.to_parquet(f"{args.data_output_path}/{args.data_output_name}", index=True)

    print(f"Evaluating performance of rate reconstruction from embeddings: {args.embeddings} and {args.rates}")

    # Interactive outlier plot (must run before reconstructed_vs_input_all_times, which modifies rates in-place)
    fig_interactive = reconstructed_vs_input_interactive(rates, reconstructed_rates, pop_labels, pop_colors, N_max)
    interactive_output_name = args.plot_output_name.replace(".png", ".html")
    fig_interactive.write_html(f"{args.plot_output_path}/{interactive_output_name}")

    # True vs Reconstructed rates across all time points
    tvr_fig = reconstructed_vs_input_all_times(rates, reconstructed_rates, N_max)
    tvr_fig.savefig(f"{args.plot_output_path}/{args.plot_output_name}", dpi=500, bbox_inches="tight")

    # Percent error in reconstructed rates
    # error = find_percent_error(rates, reconstructed_rates, N_max)
    # error.to_parquet(f"{args.data_output_path}/{args.data_output_name}", index=True)

    # Plotting reconstructed rates over time
    if len(pop_labels["population"].unique()) < 10: # only plot if there are less than 10 populations for better visualization
        rates_fig = plot_rates_multi_figs(reconstructed_rates, pop_colors, pop_labels, min_time=args.min_time, max_rate_plot=args.max_rate_plot, time_scale=args.time_scale, ylabel="Coalescence Rate")
        rates_fig_single = plot_rates_single_fig(reconstructed_rates, min_time = args.min_time, max_rate_plot = args.max_rate_plot, time_scale=args.time_scale, ylabel="Coalescence Rate")

        rates_fig.savefig(os.path.join(args.plot_output_path, args.plot_output_name.replace("performance", "reconstructed_rates")), dpi=500, bbox_inches="tight")
        rates_fig_single.savefig(os.path.join(args.plot_output_path, args.plot_output_name.replace("performance", "reconstructed_rates_single")), dpi=500, bbox_inches="tight")

if __name__ == "__main__":
    main()