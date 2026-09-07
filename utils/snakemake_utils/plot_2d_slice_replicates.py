#!/usr/bin/env python3
"""Panel of 2-D X1/X2 scatter plots at the most recent time slice, one per seed."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import svd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Panel of 2-D scatter plots at the most recent time slice"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Replicate embeddings parquet files, in seed order")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--colors", required=True, help="Population colors CSV")
    parser.add_argument("--labels", required=True, help="Population labels CSV")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--min_time", type=float, default=50,
                        help="Lower time cutoff; the most recent slice is the smallest "
                             "time >= min_time across all replicates")
    parser.add_argument("--max_time", type=float, default=2000)
    parser.add_argument("--lims", type=str, default="None",
                        help="Shared axis half-range. 'None' = auto from data.")
    parser.add_argument("--param_label", type=str, default=None)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.seeds), \
        "--inputs and --seeds must have the same number of elements"

    lims = None if args.lims == "None" else float(args.lims)

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    color_map = colors_df.set_index("population")["color"].to_dict()

    # Load and time-filter
    dfs = []
    for path in args.inputs:
        df = pd.read_parquet(path)
        df = df[(df["time"] >= args.min_time) & (df["time"] <= args.max_time)]
        dfs.append(df)

    id_col = "individual_id" if "individual_id" in dfs[0].columns else "sample_id"
    if id_col == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    pop_map = labels_df.drop_duplicates(subset=id_col).set_index(id_col)["population"]

    # Most recent = smallest time present across all replicates after filtering
    most_recent = min(df["time"].min() for df in dfs)
    print(f"Most recent time slice: {most_recent:.1f} generations")

    # Extract slices and attach population
    slices = []
    for df in dfs:
        s = df[df["time"] == most_recent].copy()
        s["population"] = s[id_col].map(pop_map).fillna("unknown")
        s["color"]      = s["population"].map(color_map).fillna("#808080")
        slices.append(s)

    # Procrustes-rotate each replicate to best match seed 0 (rotation only, no scaling)
    ref_slice = slices[0].set_index(id_col)
    common_ids = sorted(
        set.intersection(*[set(s[id_col].values) for s in slices])
    )
    ref_xy = ref_slice.loc[common_ids, ["X1", "X2"]].values
    for i in range(1, len(slices)):
        s_indexed = slices[i].set_index(id_col)
        src_xy = s_indexed.loc[common_ids, ["X1", "X2"]].values
        src_mean = src_xy.mean(axis=0)
        ref_mean = ref_xy.mean(axis=0)
        U, _, Vt = svd((src_xy - src_mean).T @ (ref_xy - ref_mean))
        R = U @ Vt  # orthogonal Procrustes: allow rotation and reflection
        all_xy = slices[i][["X1", "X2"]].values
        rotated = (all_xy - src_mean) @ R + ref_mean
        slices[i] = slices[i].copy()
        slices[i]["X1"] = rotated[:, 0]
        slices[i]["X2"] = rotated[:, 1]

    print(f"\nSamples being plotted (seed 0, t={most_recent:.0f}):")
    for _, row in slices[0].sort_values([id_col]).iterrows():
        print(f"  {id_col}={row[id_col]}  population={row['population']}  label={row['population']}-{row[id_col]}")

    # Shared axis limits
    if lims is None:
        all_x = np.concatenate([s["X1"].dropna().values for s in slices])
        all_y = np.concatenate([s["X2"].dropna().values for s in slices])
        combined = np.concatenate([all_x, all_y])
        pad  = 0.10 * (combined.max() - combined.min())
        lims = max(abs(combined.min() - pad), abs(combined.max() + pad))

    # Populations present and their legend patches
    active_samples = set(dfs[0][id_col].unique())
    active_labels  = labels_df[labels_df[id_col].isin(active_samples)]
    active_pops    = [p for p in active_labels["population"].unique() if p in color_map]
    legend_handles = [
        mpatches.Patch(facecolor=color_map[pop], label=pop)
        for pop in active_pops
    ]

    n     = len(args.seeds)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    panel_in     = 4.0
    caption_in   = 0.22 if args.param_label else 0.0
    legend_ncols = min(len(legend_handles), 8)
    legend_nrows = int(np.ceil(len(legend_handles) / legend_ncols)) if legend_handles else 1
    legend_in    = legend_nrows * 0.22 + 0.15
    fig_h = panel_in * nrows + legend_in + caption_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_in * ncols, fig_h))
    axes = np.array(axes).reshape(nrows, ncols)
    fig.subplots_adjust(hspace=0.25, wspace=0.08)

    for idx, (s, seed) in enumerate(zip(slices, args.seeds)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        ax.scatter(s["X1"], s["X2"], c=s["color"], s=12,
                   linewidths=0, zorder=5)
        label_offset = 0.015 * 2 * lims
        for _, sample in s.iterrows():
            ax.text(sample["X1"], sample["X2"] - label_offset,
                    f"{sample['population'][:2]}-{sample[id_col]}",
                    fontsize=5, ha="center", va="top",
                    color=sample["color"], fontweight="bold")

        ax.set_xlim(-lims, lims)
        ax.set_ylim(-lims, lims)
        ax.set_aspect("equal")
        ax.set_title(f"seed {seed}  |  t = {most_recent:.0f} gen",
                     fontsize=8, fontweight="bold", pad=3)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("X1", fontsize=6, labelpad=1)
        if col == 0:
            ax.set_ylabel("X2", fontsize=6, labelpad=1)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    bottom_frac  = (legend_in + caption_in) / fig_h
    legend_frac  = caption_in / fig_h

    fig.subplots_adjust(bottom=bottom_frac, hspace=0.25, wspace=0.08,
                        left=0.06, right=0.98, top=0.97)

    if args.param_label:
        fig.text(0.5, 0.005, args.param_label, ha="center", va="bottom",
                 fontsize=7, color="#555555")

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=legend_ncols,
               fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, legend_frac + 0.01))

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved 2D slice panel to {out_path}")


if __name__ == "__main__":
    main()
