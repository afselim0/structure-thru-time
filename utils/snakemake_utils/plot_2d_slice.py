#!/usr/bin/env python3
"""Single-panel 2-D X1/X2 scatter plot at the most recent time slice."""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Single-panel 2-D scatter plot at the most recent time slice"
    )
    parser.add_argument("--input", required=True, help="Embeddings parquet file")
    parser.add_argument("--colors", required=True, help="Population colors CSV")
    parser.add_argument("--labels", required=True, help="Population labels CSV")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--min_time", type=float, default=50,
                        help="Lower time cutoff; the most recent slice is the smallest "
                             "time >= min_time")
    parser.add_argument("--max_time", type=float, default=2000)
    parser.add_argument("--lims", type=str, default="None",
                        help="Axis half-range. 'None' = auto from data.")
    args = parser.parse_args()

    lims = None if args.lims == "None" else float(args.lims)

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    color_map = colors_df.set_index("population")["color"].to_dict()

    df = pd.read_parquet(args.input)
    df = df[(df["time"] >= args.min_time) & (df["time"] <= args.max_time)]

    id_col = "individual_id" if "individual_id" in df.columns else "sample_id"
    if id_col == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    pop_map = labels_df.drop_duplicates(subset=id_col).set_index(id_col)["population"]

    most_recent = df["time"].min()
    print(f"Most recent time slice: {most_recent:.1f} generations")

    s = df[df["time"] == most_recent].copy()
    s["population"] = s[id_col].map(pop_map).fillna("unknown")
    s["color"]      = s["population"].map(color_map).fillna("#808080")

    if lims is None:
        combined = np.concatenate([s["X1"].dropna().values, s["X2"].dropna().values])
        pad  = 0.10 * (combined.max() - combined.min())
        lims = max(abs(combined.min() - pad), abs(combined.max() + pad))

    active_pops = [p for p in s["population"].unique() if p in color_map]
    legend_handles = [
        mpatches.Patch(facecolor=color_map[pop], label=pop)
        for pop in active_pops
    ]
    legend_ncols = min(len(legend_handles), 8)
    legend_nrows = int(np.ceil(len(legend_handles) / legend_ncols)) if legend_handles else 1
    legend_in  = legend_nrows * 0.22 + 0.15
    panel_in   = 5.0
    fig_h      = panel_in + legend_in

    fig, ax = plt.subplots(1, 1, figsize=(panel_in, fig_h))
    fig.subplots_adjust(bottom=(legend_in / fig_h) + 0.02, left=0.10, right=0.97, top=0.93)

    ax.scatter(s["X1"], s["X2"], c=s["color"], s=18, linewidths=0, zorder=5)

    label_offset = 0.015 * 2 * lims
    for _, sample in s.iterrows():
        ax.text(sample["X1"], sample["X2"] - label_offset,
                f"{sample['population']}-{sample[id_col]}",
                fontsize=5, ha="center", va="top",
                color=sample["color"], fontweight="bold")

    ax.set_xlim(-lims, lims)
    ax.set_ylim(-lims, lims)
    ax.set_aspect("equal")
    ax.set_title(f"t = {most_recent:.0f} gen", fontsize=10, fontweight="bold", pad=4)
    ax.set_xlabel("X1", fontsize=8)
    ax.set_ylabel("X2", fontsize=8)

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=legend_ncols, fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved 2D slice to {out_path}")


if __name__ == "__main__":
    main()
