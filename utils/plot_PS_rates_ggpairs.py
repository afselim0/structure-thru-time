#!/usr/bin/env python3
"""ggpairs-style comparison of reconstructed rates across SINGER posterior sample runs.

Each panel (row=PSi, col=PSj) scatters log-reconstructed-rates from PSi (y) vs
PSj (x), with log r² annotations.  Diagonal shows the PS label.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from evaluate_performance import compute_distance_df, reconstruct_rates

MAX_PTS = 2000


def _discover_embeddings(ps_data_dir):
    matches = sorted(
        m for m in glob.glob(os.path.join(ps_data_dir, "embeddings.*.parquet"))
        if "loss" not in os.path.basename(m)
    )
    if not matches:
        raise FileNotFoundError(f"No embeddings*.parquet in {ps_data_dir}")
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime)
        print(f"  [warn] multiple embeddings in {ps_data_dir}; using newest")
    return matches[-1]


def _compute_all_rec_rates(input_paths, ps_indices, N_min, link_function, id_label):
    rates = {}
    for path, k in zip(input_paths, ps_indices):
        print(f"Computing reconstructed rates for PS{k}...")
        emb       = pd.read_parquet(path)
        r         = reconstruct_rates(compute_distance_df(emb, id_label=id_label), N_min, link_function)
        rates[k]  = r
    return rates


def _align_rates(rates_dict):
    common_pairs = None
    common_times = None
    for r in rates_dict.values():
        pairs = r.index
        times = set(r.columns.astype(float))
        common_pairs = pairs if common_pairs is None else common_pairs.intersection(pairs)
        common_times = times if common_times is None else common_times & times
    common_times = sorted(common_times)
    return {k: r.loc[common_pairs, common_times] for k, r in rates_dict.items()}


def _precompute_colors(pairs, pop_labels, pop_colors):
    _id_col   = "individual_id" if "individual_id" in pop_labels.columns else "sample_id"
    pop_map   = pop_labels.drop_duplicates(subset=_id_col).set_index(_id_col)["population"]
    color_map = pop_colors.set_index("population")["color"]

    pop1 = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2 = np.array([pop_map.get(p[1], "unknown") for p in pairs])

    facecolors, edgecolors, linewidths = [], [], []
    for p1, p2 in zip(pop1, pop2):
        key = sorted([p1, p2])
        facecolors.append(color_map.get(key[0], "#808080"))
        edgecolors.append(color_map.get(key[1], "#808080"))
        linewidths.append(0.0 if p1 == p2 else 0.5)

    unique_keys = list(dict.fromkeys(tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)))
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
        description="ggpairs of reconstructed rates across PS runs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_dir",
                       help="Directory containing PS0/, PS1/, … subdirs")
    group.add_argument("--inputs", nargs="+",
                       help="Embeddings parquet files in PS order")

    parser.add_argument("--n_ps",  type=int, default=8)
    parser.add_argument("--colors", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--N_min",  type=float, required=True)
    parser.add_argument("--N_max",  type=float, required=True)
    parser.add_argument("--link_function", type=str, default="inverse_distance")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--param_label", type=str, default=None)
    args = parser.parse_args()

    if args.inputs:
        input_paths = args.inputs
        ps_indices  = list(range(len(input_paths)))
    else:
        ps_indices  = list(range(args.n_ps))
        input_paths = []
        for k in ps_indices:
            p = _discover_embeddings(os.path.join(args.base_dir, f"chimps_PS{k}", "data"))
            input_paths.append(p)
            print(f"PS{k}: {os.path.basename(p)}")

    n = len(ps_indices)

    pop_colors_df = pd.read_csv(args.colors)
    pop_labels_df = pd.read_csv(args.labels)

    _first = pd.read_parquet(input_paths[0])
    id_label = "individual_id" if "individual_id" in _first.columns else "sample_id"
    if id_label == "individual_id" and "individual_id" not in pop_labels_df.columns:
        pop_labels_df = pop_labels_df.copy()
        pop_labels_df["individual_id"] = pop_labels_df["sample_id"] // 2

    rates_dict = _compute_all_rec_rates(input_paths, ps_indices, args.N_min,
                                        args.link_function, id_label)
    rates_dict = _align_rates(rates_dict)

    ref_k      = ps_indices[0]
    ref_rates  = rates_dict[ref_k]
    pairs      = ref_rates.index
    times      = ref_rates.columns.astype(float).values
    n_times    = len(times)

    active = set(pairs.get_level_values(0)) | set(pairs.get_level_values(1))
    pop_labels_df = pop_labels_df[pop_labels_df[id_label].isin(active)]
    pops          = pop_labels_df["population"].unique()
    pop_colors_df = pop_colors_df[pop_colors_df["population"].isin(pops)]

    facecolors, edgecolors, linewidths, pop1, pop2, legend_handles = \
        _precompute_colors(pairs, pop_labels_df, pop_colors_df)

    flat = {k: rates_dict[k].values.ravel() for k in ps_indices}

    fc_flat = np.repeat(facecolors, n_times)
    ec_flat = np.repeat(edgecolors, n_times)
    lw_flat = np.repeat(linewidths, n_times)

    all_vals = np.concatenate([v for v in flat.values()])
    valid    = np.isfinite(all_vals) & (all_vals > 0)
    rate_min = float(all_vals[valid].min())
    rate_max = float(all_vals[valid].max())
    log_lims = [np.log10(rate_min) - 0.05, np.log10(rate_max) + 0.05]

    rng     = np.random.default_rng(seed=0)
    n_pts   = len(fc_flat)
    sub_idx = rng.choice(n_pts, size=min(MAX_PTS, n_pts), replace=False)

    MAX_LEGEND_ENTRIES = 30
    MAX_LEGEND_ROWS    = 5
    display_handles    = legend_handles[:MAX_LEGEND_ENTRIES]
    n_display          = len(display_handles)
    legend_ncols_right = max(1, int(np.ceil(n_display / MAX_LEGEND_ROWS)))
    legend_w_in        = legend_ncols_right * 1.4 + 0.3
    panel_size = 1.4
    fig_w = n * panel_size + legend_w_in + 0.2
    fig_h = n * panel_size + (0.45 if args.param_label else 0)
    fig, axes = plt.subplots(n, n, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    grid_right = (n * panel_size) / fig_w
    fig.subplots_adjust(hspace=0.05, wspace=0.05, right=grid_right)

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            ki = ps_indices[row]
            kj = ps_indices[col]

            if row == col:
                ax.set_xlim(log_lims);  ax.set_ylim(log_lims)
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, f"PS{ki}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, fontweight="bold",
                        color="#333333")
                ax.set_xticks([]);  ax.set_yticks([])

            elif row < col:
                xi = flat[kj][sub_idx]
                yi = flat[ki][sub_idx]
                fc = fc_flat[sub_idx];  ec = ec_flat[sub_idx];  lw = lw_flat[sub_idx]
                vp = np.isfinite(xi) & np.isfinite(yi) & (xi > 0) & (yi > 0)
                ax.scatter(np.log10(xi[vp]), np.log10(yi[vp]),
                           c=fc[vp], edgecolors=ec[vp], linewidths=lw[vp],
                           s=2, alpha=0.3, rasterized=True)
                ax.plot(log_lims, log_lims, 'k--', lw=0.7, zorder=5)
                ax.set_xlim(log_lims);  ax.set_ylim(log_lims)
                xi_all = flat[kj];  yi_all = flat[ki]
                vv = np.isfinite(xi_all) & np.isfinite(yi_all) & (xi_all > 0) & (yi_all > 0)
                log_r2 = np.corrcoef(np.log10(xi_all[vv]), np.log10(yi_all[vv]))[0, 1] ** 2
                ax.text(0.05, 0.95, f"$r^2$={log_r2:.3f}", transform=ax.transAxes,
                        fontsize=5.5, va="top", color="#222222")

            else:
                xi_all = flat[kj];  yi_all = flat[ki]
                vv = np.isfinite(xi_all) & np.isfinite(yi_all) & (xi_all > 0) & (yi_all > 0)
                log_r2 = np.corrcoef(np.log10(xi_all[vv]), np.log10(yi_all[vv]))[0, 1] ** 2
                ax.set_facecolor("#fafafa")
                ax.text(0.5, 0.5, f"{log_r2:.3f}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=22, fontweight="bold",
                        color="#222222")
                ax.set_xticks([]);  ax.set_yticks([])
                ax.set_xlim(log_lims);  ax.set_ylim(log_lims)

            ax.tick_params(axis="both", which="both",
                           bottom=(row == n - 1) and (row != col),
                           left=(col == 0) and (row != col),
                           labelbottom=(row == n - 1) and (row != col),
                           labelleft=(col == 0) and (row != col),
                           labelsize=5, length=2, pad=1)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

            if row == n - 1 and row != col:
                ax.set_xlabel(f"PS{kj}", fontsize=6, labelpad=2)
            if col == 0 and row != col:
                ax.set_ylabel(f"PS{ki}", fontsize=6, labelpad=2)

    legend_ax = fig.add_axes(
        [(n * panel_size + 0.1) / fig_w, 0.05, legend_w_in / fig_w, 0.9]
    )
    legend_ax.axis("off")
    legend_ax.legend(handles=display_handles, loc="center left", fontsize=9,
                     title="pop pair", title_fontsize=9,
                     ncol=legend_ncols_right,
                     frameon=True, markerscale=1.2, handlelength=1.0, borderpad=0.6)

    if args.param_label:
        fig.text(0.5 * (n * panel_size) / fig_w, 0.005, args.param_label,
                 ha="center", va="bottom", fontsize=9, color="#555555")

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200)
    print(f"Saved PS rates ggpairs to {out_path}")


if __name__ == "__main__":
    main()
