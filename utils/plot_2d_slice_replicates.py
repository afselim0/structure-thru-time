#!/usr/bin/env python3
"""Panel of 2-D X1/X2 scatter plots at the most recent time slice, one per seed."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")


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
    pop_map   = labels_df.set_index("sample_id")["population"]

    # Build "POP-N" short labels: within-population rank sorted by sample_id,
    # computed once so the same sample always gets the same label across seeds.
    sample_labels: dict = {}
    for pop, grp in labels_df.groupby("population"):
        for rank, sid in enumerate(sorted(grp["sample_id"].unique()), start=1):
            sample_labels[sid] = f"{pop}-{rank}"

    # Load and time-filter
    dfs = []
    for path in args.inputs:
        df = pd.read_parquet(path)
        df = df[(df["time"] >= args.min_time) & (df["time"] <= args.max_time)]
        dfs.append(df)

    # Most recent = smallest time present across all replicates after filtering
    most_recent = min(df["time"].min() for df in dfs)
    print(f"Most recent time slice: {most_recent:.1f} generations")

    # Extract slices and attach population
    slices = []
    for df in dfs:
        s = df[df["time"] == most_recent].copy()
        s["population"] = s["sample_id"].map(pop_map).fillna("unknown")
        s["color"]      = s["population"].map(color_map).fillna("#808080")
        slices.append(s)

    # Shared axis limits
    if lims is None:
        all_x = np.concatenate([s["X1"].dropna().values for s in slices])
        all_y = np.concatenate([s["X2"].dropna().values for s in slices])
        combined = np.concatenate([all_x, all_y])
        pad  = 0.10 * (combined.max() - combined.min())
        lims = max(abs(combined.min() - pad), abs(combined.max() + pad))

    # Populations present and their legend patches
    active_samples = set(dfs[0]["sample_id"].unique())
    active_labels  = labels_df[labels_df["sample_id"].isin(active_samples)]
    active_pops    = [p for p in active_labels["population"].unique() if p in color_map]
    legend_handles = [
        mpatches.Patch(facecolor=color_map[pop], label=pop)
        for pop in active_pops
    ]

    n     = len(args.seeds)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    panel_in = 4.0
    # Bottom strip: legend (~0.5") + optional caption (~0.2")
    bottom_strip = 0.55 + (0.22 if args.param_label else 0)
    fig_h = panel_in * nrows + bottom_strip
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_in * ncols, fig_h))
    axes = np.array(axes).reshape(nrows, ncols)
    fig.subplots_adjust(hspace=0.25, wspace=0.08)

    for idx, (s, seed) in enumerate(zip(slices, args.seeds)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        for _, sample in s.iterrows():
            ax.text(sample["X1"], sample["X2"],
                    sample_labels.get(sample["sample_id"], str(sample["sample_id"])),
                    fontsize=5, ha="center", va="center",
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

    # Caption at the very bottom; legend sits above it with a clear gap
    caption_y  = 0.01
    legend_y   = (caption_y + 0.055) if args.param_label else 0.01
    bottom_pad = legend_y + 0.10   # tight_layout leaves this fraction for the legend row

    if args.param_label:
        fig.text(0.5, caption_y, args.param_label, ha="center", va="bottom",
                 fontsize=7, color="#555555")

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(len(legend_handles), 8),
               fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, legend_y))

    plt.tight_layout(rect=[0, bottom_pad, 1, 1], pad=0.4)

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved 2D slice panel to {out_path}")


if __name__ == "__main__":
    main()
