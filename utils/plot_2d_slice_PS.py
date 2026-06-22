#!/usr/bin/env python3
"""Panel of 2-D X1/X2 scatter plots at the most recent time slice, one panel per
SINGER posterior sample (PS0–PS7).  Procrustes-aligns PS1–PS7 to PS0.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import svd

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")


def _discover_embeddings(ps_data_dir):
    """Return the single embeddings parquet in ps_data_dir (not a loss file)."""
    matches = sorted(
        m for m in glob.glob(os.path.join(ps_data_dir, "embeddings.*.parquet"))
        if "loss" not in os.path.basename(m)
    )
    if not matches:
        raise FileNotFoundError(f"No embeddings*.parquet in {ps_data_dir}")
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime)
        print(f"  [warn] multiple embeddings in {ps_data_dir}; using newest: "
              f"{os.path.basename(matches[-1])}")
    return matches[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Panelled 2-D scatter across SINGER posterior samples"
    )
    # --- inputs: either explicit list OR auto-discover from base_dir ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs", nargs="+",
                       help="Embeddings parquet files, in PS order")
    group.add_argument("--base_dir",
                       help="Directory containing PS0/, PS1/, … subdirs")

    parser.add_argument("--n_ps", type=int, default=8,
                        help="Number of posterior samples (used with --base_dir)")
    parser.add_argument("--colors",  required=True, help="Population colors CSV")
    parser.add_argument("--labels",  required=True, help="Population labels CSV")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name",
                        default="2d_slice_PS_comparison.png")
    parser.add_argument("--min_time", type=float, default=50,
                        help="Smallest time >= min_time across all PS is used")
    parser.add_argument("--max_time", type=float, default=100000)
    parser.add_argument("--lims", type=str, default="None",
                        help="Shared axis half-range.  'None' = auto from data.")
    parser.add_argument("--param_label", type=str, default=None)
    args = parser.parse_args()

    lims = None if args.lims == "None" else float(args.lims)

    # Resolve input paths
    if args.inputs:
        input_paths = args.inputs
        ps_indices  = list(range(len(input_paths)))
    else:
        input_paths = []
        ps_indices  = list(range(args.n_ps))
        for k in ps_indices:
            data_dir = os.path.join(args.base_dir, f"chimps_PS{k}", "data")
            path = _discover_embeddings(data_dir)
            input_paths.append(path)
            print(f"PS{k}: {os.path.basename(path)}")

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    color_map = colors_df.set_index("population")["color"].to_dict()

    # Load & time-filter
    dfs = []
    for path in input_paths:
        df = pd.read_parquet(path)
        df = df[(df["time"] >= args.min_time) & (df["time"] <= args.max_time)]
        dfs.append(df)

    id_col = "individual_id" if "individual_id" in dfs[0].columns else "sample_id"
    if id_col == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    pop_map = labels_df.drop_duplicates(subset=id_col).set_index(id_col)["population"]

    most_recent = min(df["time"].min() for df in dfs)
    print(f"Most recent time slice: {most_recent:.1f} generations")

    slices = []
    for df in dfs:
        s = df[df["time"] == most_recent].copy()
        s["population"] = s[id_col].map(pop_map).fillna("unknown")
        s["color"]      = s["population"].map(color_map).fillna("#808080")
        slices.append(s)

    # Procrustes-rotate PS1–PS7 onto PS0 (rotation + reflection, no scaling)
    ref_slice  = slices[0].set_index(id_col)
    common_ids = sorted(set.intersection(*[set(s[id_col].values) for s in slices]))
    ref_xy     = ref_slice.loc[common_ids, ["X1", "X2"]].values
    for i in range(1, len(slices)):
        s_indexed = slices[i].set_index(id_col)
        src_xy    = s_indexed.loc[common_ids, ["X1", "X2"]].values
        src_mean  = src_xy.mean(axis=0)
        ref_mean  = ref_xy.mean(axis=0)
        U, _, Vt  = svd((src_xy - src_mean).T @ (ref_xy - ref_mean))
        R         = U @ Vt
        all_xy    = slices[i][["X1", "X2"]].values
        rotated   = (all_xy - src_mean) @ R + ref_mean
        slices[i] = slices[i].copy()
        slices[i]["X1"] = rotated[:, 0]
        slices[i]["X2"] = rotated[:, 1]

    print(f"\nSamples being plotted (PS0, t={most_recent:.0f}):")
    for _, row in slices[0].sort_values(id_col).iterrows():
        print(f"  {id_col}={row[id_col]}  population={row['population']}")

    # Shared axis limits — centered on the data, not forced to be symmetric about 0
    all_x = np.concatenate([s["X1"].dropna().values for s in slices])
    all_y = np.concatenate([s["X2"].dropna().values for s in slices])
    if lims is None:
        cx   = (all_x.max() + all_x.min()) / 2
        cy   = (all_y.max() + all_y.min()) / 2
        half = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.06
        xlim = (cx - half, cx + half)
        ylim = (cy - half, cy + half)
    else:
        xlim = (-lims, lims)
        ylim = (-lims, lims)

    active_samples = set(dfs[0][id_col].unique())
    active_labels  = labels_df[labels_df[id_col].isin(active_samples)]
    active_pops    = [p for p in active_labels["population"].unique() if p in color_map]
    legend_handles = [
        mpatches.Patch(facecolor=color_map[pop], label=pop) for pop in active_pops
    ]

    n     = len(ps_indices)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    panel_in     = 4.0
    caption_in   = 0.22 if args.param_label else 0.0
    legend_ncols = min(len(legend_handles), 8)
    legend_nrows = int(np.ceil(len(legend_handles) / legend_ncols)) if legend_handles else 1
    legend_in    = legend_nrows * 0.22 + 0.15
    fig_h        = panel_in * nrows + legend_in + caption_in

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_in * ncols, fig_h))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (s, k) in enumerate(zip(slices, ps_indices)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        ax.scatter(s["X1"], s["X2"], c=s["color"], s=12, linewidths=0, zorder=5)
        label_offset = 0.015 * (xlim[1] - xlim[0])
        for _, sample in s.iterrows():
            ax.text(sample["X1"], sample["X2"] - label_offset,
                    f"{sample['population'][:2]}-{sample[id_col]}",
                    fontsize=5, ha="center", va="top",
                    color=sample["color"], fontweight="bold")

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"PS{k}  |  t = {most_recent:.0f} gen",
                     fontsize=8, fontweight="bold", pad=3)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("X1", fontsize=6, labelpad=1)
        if col == 0:
            ax.set_ylabel("X2", fontsize=6, labelpad=1)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    bottom_frac = (legend_in + caption_in) / fig_h
    legend_frac  = caption_in / fig_h

    fig.subplots_adjust(bottom=bottom_frac, hspace=0.25, wspace=0.08,
                        left=0.06, right=0.98, top=0.97)

    if args.param_label:
        fig.text(0.5, legend_frac + 0.005, args.param_label,
                 ha="center", va="bottom", fontsize=8, style="italic")

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=legend_ncols, fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    os.makedirs(args.output_path, exist_ok=True)
    out = os.path.join(args.output_path, args.output_name)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved PS comparison panel to {out}")


if __name__ == "__main__":
    main()
