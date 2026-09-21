#!/usr/bin/env python3
"""Plot the genotype PCA produced by run_pca.py, styled like the other CoaSTT plots."""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from plotting_style import setup_plotting_style

POINT_SIZE = 18
POINT_ALPHA = 0.85
MAX_LEGEND_ENTRIES = 10


def plot_pca(pca_df, pop_labels, pop_colors, id_label="sample_id"):
    setup_plotting_style()
    pc_cols = [c for c in pca_df.columns if c.startswith("PC") and not c.endswith("_explained")]

    # Read the explained variance off the full table before any aggregation: the
    # per-individual groupby below keeps only the PC columns and would drop it.
    explained = {c: pca_df[c].iloc[0] for c in pca_df.columns if c.endswith("_explained")}

    # Per-individual plots average the two haplotype coordinates so each diploid
    # individual contributes one point, matching the embeddings plots.
    if id_label != "sample_id":
        pca_df = (
            pca_df.merge(pop_labels[["sample_id", id_label]], on="sample_id")
            .groupby(id_label, as_index=False)[pc_cols].mean()
        )

    label_cols = pop_labels[[id_label, "population"]].drop_duplicates(subset=id_label)
    merged_df = pca_df.merge(label_cols, on=id_label).merge(pop_colors, on="population")

    fig, ax = plt.subplots(figsize=(6, 6))
    # Smaller populations are drawn last so that admixed or singleton groups are
    # not hidden underneath a large tight cluster they happen to overlap.
    pops_by_size = merged_df["population"].value_counts().index[::-1]
    for z, pop in enumerate(pops_by_size):
        pop_data = merged_df[merged_df["population"] == pop]
        ax.scatter(pop_data["PC1"], pop_data["PC2"], label=pop,
                   color=pop_data["color"].iloc[0], s=POINT_SIZE, alpha=POINT_ALPHA,
                   edgecolors="none", zorder=2 + z)

    ax.set_xlabel(_axis_label("PC1", explained))
    ax.set_ylabel(_axis_label("PC2", explained))
    n_pops = merged_df["population"].nunique()
    if n_pops <= MAX_LEGEND_ENTRIES:
        ax.legend(title="Population", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig


def _axis_label(pc, explained):
    """Label a PC axis with its explained variance when run_pca.py recorded it."""
    frac = explained.get(f"{pc}_explained")
    if frac is not None and pd.notna(frac):
        return f"{pc} ({frac * 100:.1f}%)"
    return pc


def main():
    parser = argparse.ArgumentParser(description="Plot genotype PCA coordinates")
    parser.add_argument("--input", required=True, type=str, help="Path to the PCA csv written by run_pca.py")
    parser.add_argument("--colors", required=True, type=str, help="Path to the population colors csv")
    parser.add_argument("--labels", required=True, type=str, help="Path to the population labels csv")
    parser.add_argument("--output_path", required=True, type=str, help="Output directory for the plot")
    parser.add_argument("--output_name", required=True, type=str, help="Output plot filename")
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true",
                        help="If set, one point per haplotype (sample_id). Otherwise averaged per diploid individual")

    args = parser.parse_args()
    id_label = "sample_id" if args.per_haplotype else "individual_id"

    # The csv keeps sample_id as the join key; read it back as int so the merge
    # against pop_labels lines up.
    pca_df = pd.read_csv(args.input)
    labels = pd.read_csv(args.labels)
    colors = pd.read_csv(args.colors)

    fig = plot_pca(pca_df, labels, colors, id_label=id_label)
    os.makedirs(args.output_path, exist_ok=True)
    out = os.path.join(args.output_path, args.output_name)
    fig.savefig(out, dpi=500, bbox_inches="tight")
    print(f"saved PCA plot to {out}")


if __name__ == "__main__":
    main()
