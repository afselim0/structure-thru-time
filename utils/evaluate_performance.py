#%%
#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import sys
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from plot_rates import compute_median_rates

sys.path.append("../../src")
sys.path.append("../src")
from plotting_style import *
import argparse
import matplotlib.pyplot as plt
import os


def reconstructed_vs_input_all_times(input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label="sample_id", ax=None):
    # When an ax is supplied, draw onto it (for embedding in a grid); otherwise make a standalone figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    input_rates = input_rates.copy()
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates = reconstructed_rates.copy()
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    pairs = input_rates.index
    n_times = len(input_rates.columns)

    pop_map = pop_labels.drop_duplicates(subset=id_label).set_index(id_label)["population"]
    color_map = pop_colors.set_index("population")["color"]

    pop1 = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2 = np.array([pop_map.get(p[1], "unknown") for p in pairs])

    # Flatten to (n_pairs * n_times,) — one entry per (pair, time) observation
    x_flat = input_rates.values.ravel()
    y_flat = reconstructed_rates.values.ravel()
    pop1_flat = np.repeat(pop1, n_times)
    pop2_flat = np.repeat(pop2, n_times)

    # Per-point fill/border colors: fill = sorted-first-pop, border = sorted-second-pop
    facecolors, edgecolors, linewidths = [], [], []
    for p1, p2 in zip(pop1_flat, pop2_flat):
        key = sorted([p1, p2])
        ca = color_map.get(key[0], "#808080")
        cb = color_map.get(key[1], "#808080")
        facecolors.append(ca)
        edgecolors.append(cb)
        linewidths.append(0.0 if p1 == p2 else 0.6)

    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 0) & (y_flat > 0)
    x_v  = x_flat[valid]
    y_v  = y_flat[valid]
    fc_v = np.array(facecolors)[valid]
    ec_v = np.array(edgecolors)[valid]
    lw_v = np.array(linewidths)[valid]

    rng   = np.random.default_rng(seed=0)
    order = rng.permutation(len(x_v))

    lims = [min(x_v.min(), y_v.min()), max(x_v.max(), y_v.max())]
    ax.plot(lims, lims, 'k--', lw=1.5, zorder=0)
    ax.scatter(x_v[order], y_v[order], facecolors=fc_v[order], edgecolors=ec_v[order],
               linewidths=lw_v[order], alpha=0.15, s=2, zorder=1)

    # Median diamonds — collect all groups, shuffle, single scatter to avoid biased overprinting
    unique_keys = list(dict.fromkeys(tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)))
    med_xs, med_ys, med_fcs, med_ecs, med_lws, legend_handles = [], [], [], [], [], []
    for key in unique_keys:
        mask  = np.array([tuple(sorted([p1, p2])) == key for p1, p2 in zip(pop1, pop2)])
        med_x = np.nanmedian(input_rates.values[mask], axis=0)
        med_y = np.nanmedian(reconstructed_rates.values[mask], axis=0)
        vt    = np.isfinite(med_x) & np.isfinite(med_y) & (med_x > 0) & (med_y > 0)
        ca    = color_map.get(key[0], "#808080")
        cb    = color_map.get(key[1], "#808080")
        lw    = 0.0 if key[0] == key[1] else 1.0
        label = key[0] if key[0] == key[1] else f"{key[0]}–{key[1]}"
        n = int(vt.sum())
        med_xs.extend(med_x[vt]);  med_ys.extend(med_y[vt])
        med_fcs.extend([ca] * n);  med_ecs.extend([cb] * n);  med_lws.extend([lw] * n)
        legend_handles.append(Line2D([0], [0], marker='D', linestyle='none',
                                     markerfacecolor=ca, markeredgecolor=cb,
                                     markeredgewidth=lw, markersize=5, label=label))

    med_order = np.random.default_rng(seed=1).permutation(len(med_xs))
    ax.scatter(np.array(med_xs)[med_order], np.array(med_ys)[med_order],
               facecolors=np.array(med_fcs)[med_order], edgecolors=np.array(med_ecs)[med_order],
               linewidths=np.array(med_lws)[med_order], s=30, marker='D', zorder=5)

    ax.set_xlabel(r"Input Rates + $\frac{1}{N_{\text{max}}}$")
    ax.set_ylabel(r"Reconstructed Rates + $\frac{1}{N_{\text{max}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=7, markerscale=1.5,
              handlelength=1.2, borderpad=0.5, loc="lower right")

    r2     = np.corrcoef(x_v, y_v)[0, 1] ** 2
    log_r2 = np.corrcoef(np.log(x_v), np.log(y_v))[0, 1] ** 2
    corr, _ = spearmanr(x_v, y_v)
    ax.text(0.05, 0.95, f"$r^2$ = {r2:.3f}",           transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.05, 0.90, f"$r^2$ (log) = {log_r2:.3f}", transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.05, 0.85, f"Spearman $\\rho$ = {corr:.3f}", transform=ax.transAxes, fontsize=8, va='top')

    return fig

def reconstructed_vs_input_hist2d(input_rates, reconstructed_rates, N_max, ax=None, x_label="input rates",
                                   shared_lims=None, shared_vmax=None, draw_colorbar=True, ylabel=True,
                                   n_bins=None):
    # Default (cheap) performance summary: 2D log-log histogram instead of an O(n_pairs*T) scatter
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    input_rates = input_rates.copy()
    input_rates += 1/N_max
    x_flat = input_rates.values.ravel()
    y_flat = reconstructed_rates.values.ravel()

    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 0) & (y_flat > 0)
    x_v = x_flat[valid]
    y_v = y_flat[valid]
    log_x = np.log10(x_v)
    log_y = np.log10(y_v)

    # Bin count scales with sqrt(n observations) so resolution adapts to dataset size, clipped to a sane range;
    # pass n_bins explicitly to override (it is the number of grid points per row/column)
    if n_bins is None:
        n_bins = int(np.clip(np.sqrt(len(x_v)), 30, 300))
    # Shared axis limits (in log space) let multiple panels be compared on a common scale
    if shared_lims is None:
        lo = min(log_x.min(), log_y.min())
        hi = max(log_x.max(), log_y.max())
    else:
        lo, hi = np.log10(shared_lims[0]), np.log10(shared_lims[1])
    bin_edges = np.linspace(lo, hi, n_bins + 1)

    counts, xedges, yedges = np.histogram2d(log_x, log_y, bins=[bin_edges, bin_edges])
    vmax = max(counts.max(), 1) if shared_vmax is None else shared_vmax
    # rasterize: vector PDF output draws each mesh cell as a separate patch, leaving
    # visible white anti-aliasing seams between cells; rasterizing embeds it as one bitmap instead
    mesh = ax.pcolormesh(10**xedges, 10**yedges, counts.T, cmap="inferno",
                          norm=colors.SymLogNorm(linthresh=1, vmin=0, vmax=vmax),
                          rasterized=True)
    if draw_colorbar:
        fig.colorbar(mesh, ax=ax, label="Number of points (pair, timepoint)")

    lims = [10**lo, 10**hi]
    ax.plot(lims, lims, 'k--', lw=1, zorder=2, color='white')

    ax.set_xlabel(rf"{x_label} + $\frac{{1}}{{N_{{\text{{max}}}}}}$")
    if ylabel:
        ax.set_ylabel(r"Reconstructed Rates + $\frac{1}{N_{\text{max}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    r2     = np.corrcoef(x_v, y_v)[0, 1] ** 2
    log_r2 = np.corrcoef(log_x, log_y)[0, 1] ** 2
    corr, _ = spearmanr(x_v, y_v)
    stats_text = f"$r^2$ = {r2:.3f}\n$r^2$ (log) = {log_r2:.3f}\nSpearman $\\rho$ = {corr:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8, va='top', color='white',
            bbox=dict(facecolor='black', edgecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    return fig, mesh

def reconstructed_vs_input_query(sample_id, input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label="sample_id"):
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
    other_sample_labels = pop_labels.drop_duplicates(subset=id_label).set_index(id_label).loc[other_sample_ids]["population"].values
    other_sample_colors = pop_colors.set_index("population").loc[other_sample_labels]["color"].values

    filtered_input_rates = input_rates.iloc[pair_indices].values
    filtered_reconstructed_rates = reconstructed_rates.iloc[pair_indices].values

    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(filtered_input_rates.shape[0]):
        ax.scatter(
            filtered_input_rates[i, :],
            filtered_reconstructed_rates[i, :],
            alpha=0.1,
            s=1,
            color=other_sample_colors[i]
        )
    ax.set_xlabel(r"Input Rates + $\frac{1}{N_{\text{max}}}$")
    ax.set_ylabel(r"Reconstructed Rates + $\frac{1}{N_{\text{max}}}$")
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

def reconstructed_vs_input_interactive(input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label="sample_id", x_label="input rate"):
    # Median input vs median reconstructed rate per population pair, one point per time point.
    # Avoids the O(n_pairs*T) point cloud — collapses each pair-of-populations to K*(K+1)/2 trajectories.
    input_rates = input_rates.copy()
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates = reconstructed_rates.copy()
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    per_haplotype = (id_label == "sample_id")
    times = input_rates.columns.values
    color_map = pop_colors.set_index("population")["color"]

    med_input, populations = compute_median_rates(input_rates, pop_labels, per_haplotype=per_haplotype)
    med_reconstructed, _    = compute_median_rates(reconstructed_rates, pop_labels, per_haplotype=per_haplotype)

    # Capture full-data axis limits from the underlying pairwise rates (not just the medians)
    all_vals = np.concatenate([input_rates.values.ravel(), reconstructed_rates.values.ravel()])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    rate_min = float(all_vals.min())
    rate_max = float(all_vals.max())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[rate_min, rate_max], y=[rate_min, rate_max],
        mode="lines", line=dict(color="black", dash="dash", width=2),
        name="y = x", showlegend=True,
    ))

    K = len(populations)
    for i in range(K):
        for j in range(i, K):
            pop_i, pop_j = populations[i], populations[j]
            med_x, med_y = med_input[i, j, :], med_reconstructed[i, j, :]
            valid = np.isfinite(med_x) & np.isfinite(med_y) & (med_x > 0) & (med_y > 0)
            if valid.sum() == 0:
                continue

            ca    = color_map.get(pop_i, "#808080")
            cb    = color_map.get(pop_j, "#808080")
            lw    = 1.5 if pop_i != pop_j else 0
            label = pop_i if pop_i == pop_j else f"{pop_i}–{pop_j}"

            fig.add_trace(go.Scatter(
                x=med_x[valid], y=med_y[valid],
                mode="markers+lines",
                marker=dict(size=8, color=ca, opacity=0.9, line=dict(color=cb, width=lw)),
                line=dict(color=ca, width=1),
                customdata=times[valid].reshape(-1, 1),
                name=label, legendgroup=label,
                hovertemplate=(
                    f"<b>{label} (median)</b><br>"
                    "Time: %{customdata[0]:.1f}<br>"
                    f"Median {x_label}: " "%{x:.3e}<br>"
                    "Median reconstructed rate: %{y:.3e}"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=f"Median {x_label} vs. reconstructed rate per population pair, over time",
        xaxis=dict(title=f"Median {x_label} + 1/N_max", type="log",
                   range=[math.log10(rate_min), math.log10(rate_max)], exponentformat="e"),
        yaxis=dict(title="Median reconstructed rate",   type="log",
                   range=[math.log10(rate_min), math.log10(rate_max)], exponentformat="e"),
    )
    return fig


def align_time_columns(input_rates, reconstructed_rates, label):
    """Align input_rates onto reconstructed_rates' time grid.

    input_rates and reconstructed_rates may come from independently generated np.linspace
    grids over max_time (with n_timepoints potentially differing between the two) that rarely
    share exact float values. Each
    reconstructed_rates timepoint is matched to its nearest input_rates timepoint instead of
    requiring an exact match.
    """
    input_times = input_rates.columns.values.astype(float)
    reconstructed_times = reconstructed_rates.columns.values.astype(float)
    if np.array_equal(input_times, reconstructed_times):
        return input_rates, reconstructed_rates

    nearest_idx = np.searchsorted(input_times, reconstructed_times)
    nearest_idx = np.clip(nearest_idx, 1, len(input_times) - 1)
    left, right = nearest_idx - 1, nearest_idx
    nearest_idx = np.where(
        np.abs(reconstructed_times - input_times[left]) <= np.abs(reconstructed_times - input_times[right]),
        left, right,
    )
    matched_input_times = input_times[nearest_idx]

    median_spacing = np.median(np.diff(np.sort(input_times)))
    max_gap = np.abs(matched_input_times - reconstructed_times)
    within_tol = max_gap <= median_spacing
    if not within_tol.all():
        print(f"  [warn] {(~within_tol).sum()} reconstructed_rates timepoints have no {label} rates "
              f"timepoint within {median_spacing:.3g} generations — dropping")

    reconstructed_rates = reconstructed_rates.loc[:, within_tol]
    matched_input_times = matched_input_times[within_tol]
    input_rates = input_rates[matched_input_times]
    input_rates.columns = reconstructed_rates.columns  # align column labels for element-wise comparison
    return input_rates, reconstructed_rates


def evaluate_and_plot(input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label,
                       plot_output_path, hist2d_output_name, interactive_output_name, label):
    """Align input_rates to reconstructed_rates, then render the standard hist2d + interactive comparison plots."""
    # Align to common timepoints (input_rates and reconstructed_rates may use different time grids)
    input_rates, reconstructed_rates = align_time_columns(input_rates, reconstructed_rates, label)

    # Align to common pairs (individual-level rates and reconstructed may differ)
    common_pairs = input_rates.index.intersection(reconstructed_rates.index)
    if len(common_pairs) < len(input_rates):
        print(f"  [warn] {len(input_rates) - len(common_pairs)} pairs in {label} rates not in reconstructed_rates — dropping")
    if len(common_pairs) < len(reconstructed_rates):
        print(f"  [warn] {len(reconstructed_rates) - len(common_pairs)} pairs in reconstructed_rates not in {label} rates — dropping")
    input_rates = input_rates.loc[common_pairs]
    reconstructed_rates = reconstructed_rates.loc[common_pairs]

    # Filter pop_labels to the individuals actually present in the data
    active_ids = np.unique([s for pair in input_rates.index for s in pair])
    pop_labels = pop_labels[pop_labels[id_label].isin(active_ids)].copy()
    pop_colors = pop_colors[pop_colors["population"].isin(pop_labels["population"].unique())]

    x_label = f"{label} rates"

    # Standard output: cheap 2D log-log histogram of input vs reconstructed rates
    hist_fig, _ = reconstructed_vs_input_hist2d(input_rates, reconstructed_rates, N_max, x_label=x_label)
    hist_fig.savefig(f"{plot_output_path}/{hist2d_output_name}", dpi=500, bbox_inches="tight")

    # Standard output: interactive median input vs reconstructed rate per population pair, over time
    fig_interactive = reconstructed_vs_input_interactive(input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label=id_label, x_label=x_label)
    fig_interactive.write_html(f"{plot_output_path}/{interactive_output_name}")


def main():
    argparser = argparse.ArgumentParser(description="Evaluate performance of rate reconstruction from embeddings")
    argparser.add_argument("--rates", type=str, required=True, help="Path to input (estimated or true) rates parquet file")
    argparser.add_argument("--reconstructed_rates", type=str, required=True, help="Path to reconstructed rates parquet file (computed from embeddings by compute_embeddings.py)")
    argparser.add_argument("--true_rates", type=str, default=None, help="Path to ground-truth demographic rates parquet file (from compute_rates_true.py), if available. Produces an additional true_rates-vs-reconstructed comparison.")
    argparser.add_argument("--colors", type=str, required=True, help="Path to population colors csv file")
    argparser.add_argument("--labels", type=str, required=True, help="Path to population labels csv file")
    argparser.add_argument("--N_max", type=float, required=True, help="N_max used to compute embeddings, sets the plotted rate floor (1/N_max)")
    argparser.add_argument("--plot_output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--hist2d_output_name", type=str, required=True, help="Filename for the input-vs-reconstructed 2D histogram png")
    argparser.add_argument("--interactive_output_name", type=str, required=True, help="Filename for the input-vs-reconstructed median interactive html")
    argparser.add_argument("--true_hist2d_output_name", type=str, default=None, help="Filename for the true-vs-reconstructed 2D histogram png (required if --true_rates is given)")
    argparser.add_argument("--true_interactive_output_name", type=str, default=None, help="Filename for the true-vs-reconstructed median interactive html (required if --true_rates is given)")
    argparser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true", help="If set, rates are per-haplotype (sample_id). Otherwise they are per-individual (individual_id).")
    args = argparser.parse_args()

    if args.true_rates is not None and (args.true_hist2d_output_name is None or args.true_interactive_output_name is None):
        raise ValueError("--true_hist2d_output_name and --true_interactive_output_name are required when --true_rates is given")

    # Loading data
    per_haplotype = args.per_haplotype
    id_label = "sample_id" if per_haplotype else "individual_id"
    N_max = args.N_max
    rates = pd.read_parquet(args.rates)
    rates.columns = rates.columns.astype(float)
    pop_colors = pd.read_csv(args.colors)
    pop_labels = pd.read_csv(args.labels)
    if not per_haplotype and "individual_id" not in pop_labels.columns:
        pop_labels = pop_labels.copy()
        pop_labels["individual_id"] = pop_labels["sample_id"] // 2

    reconstructed_rates = pd.read_parquet(args.reconstructed_rates)
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    print(f"Evaluating performance: {args.rates}")

    # Standard output: input (estimated) rates vs reconstructed rates
    evaluate_and_plot(rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label,
                       args.plot_output_path, args.hist2d_output_name, args.interactive_output_name,
                       label="input")

    # Additional output, only when ground-truth demographic rates are available: true rates vs reconstructed rates
    if args.true_rates is not None:
        true_rates = pd.read_parquet(args.true_rates)
        true_rates.columns = true_rates.columns.astype(float)
        print(f"Evaluating performance against true rates: {args.true_rates}")
        evaluate_and_plot(true_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label,
                           args.plot_output_path, args.true_hist2d_output_name, args.true_interactive_output_name,
                           label="true")

if __name__ == "__main__":
    main()
# %%
