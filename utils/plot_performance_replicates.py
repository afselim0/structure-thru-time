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

from evaluate_performance import compute_distance_df, reconstruct_rates


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
    pop_map   = pop_labels.set_index("sample_id")["population"]
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
    args = parser.parse_args()

    assert len(args.inputs) == len(args.seeds), \
        "--inputs and --seeds must have the same number of elements"

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    rates_df  = pd.read_parquet(args.rates)
    rates_df.columns = rates_df.columns.astype(float)

    n     = len(args.seeds)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    # Reserve a fixed bottom strip: legend row + optional caption
    caption_in = 0.32 if args.param_label else 0.0
    legend_in  = 0.65
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

        emb      = pd.read_parquet(path)
        rec      = reconstruct_rates(compute_distance_df(emb),
                                     args.N_min, args.link_function)
        handles  = _perf_on_ax(ax, rates_df, rec, labels_df, colors_df, args.N_max, seed)
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
                 fontsize=8, color="#555555")

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 5),
                   fontsize=9, title="pop pair", title_fontsize=9,
                   frameon=True, framealpha=0.9,
                   bbox_to_anchor=(0.5, caption_frac + 0.01))

    plt.tight_layout(rect=[0, legend_frac, 1, 1], pad=0.4)
    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved performance panel to {out_path}")


if __name__ == "__main__":
    main()
