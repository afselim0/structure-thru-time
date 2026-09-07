#!/usr/bin/env python3
"""8-panel reconstructed-vs-input scatter, one panel per replicate seed."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from OptimizationFunctions import compute_distance_df, reconstruct_rates


def _perf_on_ax(ax, input_rates, rec_rates, pop_labels, pop_colors, N_max, seed):
    """Draw a reconstructed-vs-input scatter onto an existing Axes."""
    inp = input_rates.copy()
    inp.columns = inp.columns.astype(float)
    rec = rec_rates.copy()
    rec.columns = rec.columns.astype(float)

    # Align to common timepoints
    common_times = sorted(set(inp.columns) & set(rec.columns))
    inp = inp[common_times]
    rec = rec[common_times]

    # Align to common pairs (index values, ignoring name differences)
    inp.index.names = ["s1", "s2"]
    rec.index.names = ["s1", "s2"]
    common_pairs = inp.index.intersection(rec.index)
    inp = inp.loc[common_pairs]
    rec = rec.loc[common_pairs]

    inp += 1.0 / N_max

    pairs   = inp.index
    n_times = len(common_times)
    _id_col   = "individual_id" if "individual_id" in pop_labels.columns else "sample_id"
    pop_map   = pop_labels.drop_duplicates(subset=_id_col).set_index(_id_col)["population"]
    color_map = pop_colors.set_index("population")["color"]
    pop1 = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2 = np.array([pop_map.get(p[1], "unknown") for p in pairs])

    fc = np.array([color_map.get(sorted([p1, p2])[0], "#808080")
                   for p1, p2 in zip(pop1, pop2)])
    fc_flat = np.repeat(fc, n_times)

    x_flat = inp.values.ravel()
    y_flat = rec.values.ravel()
    valid  = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 0) & (y_flat > 0)
    x_v, y_v = x_flat[valid], y_flat[valid]
    fc_v     = fc_flat[valid]

    lims = [min(x_v.min(), y_v.min()), max(x_v.max(), y_v.max())]
    ax.plot(lims, lims, "k--", lw=0.8, zorder=0)

    order = np.random.default_rng(seed=0).permutation(len(x_v))
    ax.scatter(x_v[order], y_v[order], c=fc_v[order],
               alpha=0.12, s=2, zorder=1, rasterized=True, linewidths=0)

    # Median diamonds per pop-pair type
    unique_keys = list(dict.fromkeys(tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)))
    med_xs, med_ys, med_fcs = [], [], []
    legend_handles = []
    for key in unique_keys:
        mask  = np.array([tuple(sorted([p1, p2])) == key for p1, p2 in zip(pop1, pop2)])
        mx    = np.nanmedian(inp.values[mask], axis=0)
        my    = np.nanmedian(rec.values[mask], axis=0)
        vt    = np.isfinite(mx) & np.isfinite(my) & (mx > 0) & (my > 0)
        ca    = color_map.get(key[0], "#808080")
        cb    = color_map.get(key[1], "#808080")
        lw    = 0.0 if key[0] == key[1] else 0.8
        label = key[0] if key[0] == key[1] else f"{key[0]}–{key[1]}"
        med_xs.extend(mx[vt]);  med_ys.extend(my[vt]);  med_fcs.extend([ca] * vt.sum())
        legend_handles.append(
            Line2D([0], [0], marker="D", linestyle="none",
                   markerfacecolor=ca, markeredgecolor=cb,
                   markeredgewidth=lw, markersize=7, label=label)
        )
    if med_xs:
        mo = np.random.default_rng(seed=1).permutation(len(med_xs))
        ax.scatter(np.array(med_xs)[mo], np.array(med_ys)[mo],
                   c=np.array(med_fcs)[mo], s=35, marker="D", zorder=5,
                   linewidths=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(labelsize=8)

    log_r2 = np.corrcoef(np.log(x_v), np.log(y_v))[0, 1] ** 2
    ax.text(0.05, 0.95, f"$r^2$ (log) = {log_r2:.3f}",
            transform=ax.transAxes, fontsize=9, va="top", color="#222222")
    ax.set_title(f"seed {seed}", fontsize=10, fontweight="bold", pad=3)

    return legend_handles


def main():
    parser = argparse.ArgumentParser(
        description="Panel of reconstructed-vs-input scatter plots, one per seed"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Replicate embeddings parquet files, in seed order")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--rates", required=True, help="Input rates parquet")
    parser.add_argument("--colors", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--N_min", type=float, required=True)
    parser.add_argument("--N_max", type=float, required=True)
    parser.add_argument("--link_function", type=str, default="inverse_distance")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--param_label", type=str, default=None)
    parser.add_argument("--max_time_points", type=int, default=None,
                        help="Subsample to at most this many evenly-spaced time points")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Subsample to at most this many individual pairs")
    args = parser.parse_args()

    assert len(args.inputs) == len(args.seeds), \
        "--inputs and --seeds must have the same number of elements"

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    rates_df  = pd.read_parquet(args.rates)
    rates_df.columns = rates_df.columns.astype(float)

    _first_emb = pd.read_parquet(args.inputs[0])
    id_label   = "individual_id" if "individual_id" in _first_emb.columns else "sample_id"
    if id_label == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2

    # Restrict labels to samples present in the data (labels file may cover all populations)
    active_samples = set(_first_emb[id_label].unique())
    labels_df = labels_df[labels_df[id_label].isin(active_samples)]

    # Optional subsetting of time points
    if args.max_time_points is not None:
        all_times = sorted(rates_df.columns.astype(float).tolist())
        if len(all_times) > args.max_time_points:
            idx_keep = np.round(np.linspace(0, len(all_times) - 1,
                                            args.max_time_points)).astype(int)
            keep_times = [all_times[i] for i in idx_keep]
            rates_df = rates_df[[c for c in rates_df.columns
                                  if float(c) in set(keep_times)]]
            print(f"Subsampled to {len(keep_times)} time points")

    # Optional subsetting of pairs
    pair_subsample_idx = None
    if args.max_pairs is not None:
        all_pairs = rates_df.index.tolist()
        if len(all_pairs) > args.max_pairs:
            rng = np.random.default_rng(seed=42)
            pair_subsample_idx = rng.choice(len(all_pairs), size=args.max_pairs,
                                            replace=False)
            print(f"Subsampled to {args.max_pairs} pairs (from {len(all_pairs)})")

    n     = len(args.seeds)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    # Size legend strip — cap at MAX_LEGEND_ROWS to prevent blowup with many pops
    MAX_LEGEND_ROWS = 4
    caption_in   = 0.32 if args.param_label else 0.0
    n_pops       = labels_df["population"].nunique()
    n_pairs_est  = n_pops * (n_pops + 1) // 2
    legend_ncols = max(5, int(np.ceil(n_pairs_est / MAX_LEGEND_ROWS)))
    legend_nrows = min(MAX_LEGEND_ROWS,
                       int(np.ceil(n_pairs_est / legend_ncols)) if n_pairs_est else 1)
    legend_in    = legend_nrows * 0.22 + 0.20
    panel_w, panel_h = 3.8, 3.2
    fig_h = panel_h * nrows + legend_in + caption_in

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, fig_h),
                             squeeze=False)
    fig.subplots_adjust(hspace=0.32, wspace=0.22)

    legend_handles = None
    for idx, (path, seed) in enumerate(zip(args.inputs, args.seeds)):
        print(f"Processing seed {seed}...")
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        emb = pd.read_parquet(path)
        rec = reconstruct_rates(compute_distance_df(emb, id_label=id_label),
                                args.N_min, link=args.link_function)

        # Apply pair subsample to both rates and reconstructed
        inp_plot = rates_df
        rec_plot = rec
        if pair_subsample_idx is not None:
            all_pairs = rates_df.index.tolist()
            keep = [all_pairs[i] for i in pair_subsample_idx
                    if all_pairs[i] in rec.index]
            inp_plot = rates_df.loc[keep]
            rec_plot = rec.loc[[p for p in keep if p in rec.index]]

        handles = _perf_on_ax(ax, inp_plot, rec_plot, labels_df, colors_df, args.N_max, seed)
        if legend_handles is None:
            legend_handles = handles

        if row == nrows - 1:
            ax.set_xlabel("input + 1/N_max", fontsize=9, labelpad=3)
        if col == 0:
            ax.set_ylabel("reconstructed", fontsize=9, labelpad=3)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    # Caption at very bottom, legend clearly above it
    caption_frac = caption_in / fig_h
    legend_frac  = (legend_in + caption_in) / fig_h

    if args.param_label:
        fig.text(0.5, 0.005, args.param_label, ha="center", va="bottom",
                 fontsize=9, color="#555555")

    if legend_handles:
        # Limit displayed legend entries to what fits in MAX_LEGEND_ROWS rows
        display_handles = legend_handles[:legend_ncols * MAX_LEGEND_ROWS]
        fig.legend(handles=display_handles, loc="lower center",
                   ncol=legend_ncols,
                   fontsize=9, title="pop pair", title_fontsize=9,
                   frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, caption_frac + 0.01))

    fig.subplots_adjust(bottom=legend_frac, hspace=0.32, wspace=0.22)
    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200)
    print(f"Saved performance panel to {out_path}")


if __name__ == "__main__":
    main()
