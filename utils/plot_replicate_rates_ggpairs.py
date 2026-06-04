#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from evaluate_performance import compute_distance_df, reconstruct_rates


# Max points to scatter per upper-triangle panel (for speed)
MAX_PTS = 2000


def _compute_all_rates(inputs, seeds, N_min, link_function):
    """Load embeddings and compute reconstructed rates for every seed."""
    rates = {}
    for path, seed in zip(inputs, seeds):
        print(f"Computing reconstructed rates for seed {seed}...")
        emb = pd.read_parquet(path)
        r   = reconstruct_rates(compute_distance_df(emb), N_min, link_function)
        rates[seed] = r
    return rates


def _align_rates(rates_dict):
    """Restrict all rate DataFrames to the intersection of pairs and timepoints."""
    common_pairs = None
    common_times = None
    for r in rates_dict.values():
        pairs = r.index
        times = set(r.columns.astype(float))
        common_pairs = pairs if common_pairs is None else common_pairs.intersection(pairs)
        common_times = times if common_times is None else common_times & times
    common_times = sorted(common_times)
    return {s: r.loc[common_pairs, common_times] for s, r in rates_dict.items()}


def _precompute_colors(pairs, pop_labels, pop_colors):
    """Return facecolors, edgecolors, linewidths, and legend handles for every pair."""
    pop_map   = pop_labels.set_index("sample_id")["population"]
    color_map = pop_colors.set_index("population")["color"]

    pop1 = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2 = np.array([pop_map.get(p[1], "unknown") for p in pairs])

    facecolors, edgecolors, linewidths = [], [], []
    for p1, p2 in zip(pop1, pop2):
        key = sorted([p1, p2])
        facecolors.append(color_map.get(key[0], "#808080"))
        edgecolors.append(color_map.get(key[1], "#808080"))
        linewidths.append(0.0 if p1 == p2 else 0.5)

    # Legend entries — one per unique population-pair type
    unique_keys  = list(dict.fromkeys(tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)))
    legend_handles = []
    for key in unique_keys:
        ca    = color_map.get(key[0], "#808080")
        cb    = color_map.get(key[1], "#808080")
        lw    = 0.0 if key[0] == key[1] else 0.8
        label = key[0] if key[0] == key[1] else f"{key[0]}–{key[1]}"
        legend_handles.append(Line2D([0], [0], marker='o', linestyle='none',
                                     markerfacecolor=ca, markeredgecolor=cb,
                                     markeredgewidth=lw, markersize=5, label=label))
    return (np.array(facecolors), np.array(edgecolors),
            np.array(linewidths), pop1, pop2, legend_handles)


def main():
    parser = argparse.ArgumentParser(
        description="ggpairs-style pairwise comparison of reconstructed rates across seeds"
    )
    parser.add_argument("--inputs",  nargs="+", required=True,
                        help="Embeddings parquet files in seed order")
    parser.add_argument("--seeds",   nargs="+", type=int, required=True)
    parser.add_argument("--colors",  required=True)
    parser.add_argument("--labels",  required=True)
    parser.add_argument("--N_min",   type=float, required=True)
    parser.add_argument("--N_max",   type=float, required=True)
    parser.add_argument("--link_function", type=str, default="inverse_distance")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--param_label", type=str, default=None)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.seeds)
    n = len(args.seeds)

    pop_colors_df = pd.read_csv(args.colors)
    pop_labels_df = pd.read_csv(args.labels)

    # Compute and align rates
    rates_dict = _compute_all_rates(args.inputs, args.seeds, args.N_min, args.link_function)
    rates_dict = _align_rates(rates_dict)

    # Reference seed for pair/time structure
    ref_seed  = args.seeds[0]
    ref_rates = rates_dict[ref_seed]
    pairs     = ref_rates.index
    times     = ref_rates.columns.astype(float).values
    n_times   = len(times)

    # Subset labels/colors to active samples
    active = set(pairs.get_level_values(0)) | set(pairs.get_level_values(1))
    pop_labels_df = pop_labels_df[pop_labels_df["sample_id"].isin(active)]
    pops          = pop_labels_df["population"].unique()
    pop_colors_df = pop_colors_df[pop_colors_df["population"].isin(pops)]

    facecolors, edgecolors, linewidths, pop1, pop2, legend_handles = \
        _precompute_colors(pairs, pop_labels_df, pop_colors_df)

    # Flatten rates for each seed: shape (n_pairs * n_times,)
    flat = {s: rates_dict[s].values.ravel() for s in args.seeds}

    # Repeat per-pair colors over timepoints
    fc_flat = np.repeat(facecolors, n_times)
    ec_flat = np.repeat(edgecolors, n_times)
    lw_flat = np.repeat(linewidths, n_times)

    # Global axis range from all seeds combined
    all_vals = np.concatenate([v for v in flat.values()])
    valid    = np.isfinite(all_vals) & (all_vals > 0)
    rate_min = float(all_vals[valid].min())
    rate_max = float(all_vals[valid].max())
    log_lims = [np.log10(rate_min) - 0.05, np.log10(rate_max) + 0.05]

    # Fixed subsample indices for consistency across all panels
    rng          = np.random.default_rng(seed=0)
    n_pts        = len(fc_flat)
    sub_idx      = rng.choice(n_pts, size=min(MAX_PTS, n_pts), replace=False)

    # Figure: n×n grid + room for legend on right
    panel_size = 1.4
    fig_w = n * panel_size + 1.8   # extra for legend
    fig_h = n * panel_size + (0.35 if args.param_label else 0)
    fig, axes = plt.subplots(n, n, figsize=(fig_w, fig_h),
                             sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05,
                        right=(n * panel_size) / fig_w)

    for row in range(n):
        for col in range(n):
            ax  = axes[row, col]
            si  = args.seeds[row]
            sj  = args.seeds[col]

            if row == col:
                # Diagonal: seed label
                ax.set_xlim(log_lims); ax.set_ylim(log_lims)
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, f"seed {si}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, fontweight="bold", color="#333333")
                ax.set_xticks([]); ax.set_yticks([])

            elif row < col:
                # Upper triangle: scatter seed_j (x) vs seed_i (y)  — seed_j is the col seed
                xi = flat[sj][sub_idx]   # x = column seed
                yi = flat[si][sub_idx]   # y = row seed
                fc = fc_flat[sub_idx]
                ec = ec_flat[sub_idx]
                lw = lw_flat[sub_idx]
                valid_pt = np.isfinite(xi) & np.isfinite(yi) & (xi > 0) & (yi > 0)
                ax.scatter(np.log10(xi[valid_pt]), np.log10(yi[valid_pt]),
                           c=fc[valid_pt], edgecolors=ec[valid_pt],
                           linewidths=lw[valid_pt], s=2, alpha=0.3, rasterized=True)
                ax.plot(log_lims, log_lims, 'k--', lw=0.7, zorder=5)
                ax.set_xlim(log_lims); ax.set_ylim(log_lims)

                # log r² in corner
                xi_all = flat[sj]; yi_all = flat[si]
                vv = np.isfinite(xi_all) & np.isfinite(yi_all) & (xi_all > 0) & (yi_all > 0)
                log_r2 = np.corrcoef(np.log10(xi_all[vv]), np.log10(yi_all[vv]))[0, 1] ** 2
                ax.text(0.05, 0.95, f"$r^2$={log_r2:.3f}", transform=ax.transAxes,
                        fontsize=5.5, va="top", color="#222222")

            else:
                # Lower triangle: r² as large centered text
                xi_all = flat[sj]; yi_all = flat[si]
                vv = np.isfinite(xi_all) & np.isfinite(yi_all) & (xi_all > 0) & (yi_all > 0)
                log_r2 = np.corrcoef(np.log10(xi_all[vv]), np.log10(yi_all[vv]))[0, 1] ** 2
                ax.set_facecolor("#fafafa")
                ax.text(0.5, 0.5, f"{log_r2:.3f}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#222222")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(log_lims); ax.set_ylim(log_lims)

            # Tick/spine cleanup: only show ticks on outer edges
            show_bottom = (row == n - 1) and (col != col)   # never — handled below
            ax.tick_params(axis="both", which="both",
                           bottom=(row == n - 1) and (row != col),
                           left=(col == 0) and (row != col),
                           labelbottom=(row == n - 1) and (row != col),
                           labelleft=(col == 0) and (row != col),
                           labelsize=5, length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

            # Axis labels on outer edges only
            if row == n - 1 and row != col:
                ax.set_xlabel(f"seed {sj}", fontsize=6, labelpad=2)
            if col == 0 and row != col:
                ax.set_ylabel(f"seed {si}", fontsize=6, labelpad=2)

    # Shared legend to the right
    legend_ax = fig.add_axes([(n * panel_size + 0.1) / fig_w, 0.1, 1.5 / fig_w, 0.8])
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_handles, loc="center left", fontsize=7,
                     title="pop pair", title_fontsize=7,
                     frameon=True, markerscale=1.2, handlelength=1.0, borderpad=0.6)

    # Subtitle
    if args.param_label:
        fig.text(0.5 * (n * panel_size) / fig_w, 0.01, args.param_label,
                 ha="center", va="bottom", fontsize=7, color="#555555")

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved ggpairs plot to {out_path}")


if __name__ == "__main__":
    main()
