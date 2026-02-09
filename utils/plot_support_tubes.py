#!/usr/bin/env python3
"""
utils/plot_support_tubes.py

CLI wrapper for generating support tube visualizations.
Produces an HTML plot with sample trajectories, support tube meshes,
and ring outlines for each population.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Add repo root and src/ to path so PlotEmbeddings and utils.support_tube resolve
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from PlotEmbeddings import PlotEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Plot embeddings with support tubes")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors CSV file")
    parser.add_argument("--labels", required=True, help="Path to population labels CSV file")
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    parser.add_argument("--min_time", default=0, type=float, help="Minimum time for plotting")
    parser.add_argument("--generation_time", default=None, type=float, help="Generation time in years")

    # Support tube parameters (primary knobs)
    parser.add_argument("--grid_res_xy", default=256, type=int, help="XY grid resolution")
    parser.add_argument("--grid_res_u", default=256, type=int, help="Number of time slices")
    parser.add_argument("--mass_frac", default=0.90, type=float, help="Fraction of population mass captured")
    parser.add_argument("--sigma_u", default=1.0, type=float, help="Temporal smoothing along time axis")

    args = parser.parse_args()

    print("Reading files...")
    embeddings_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)

    # Subset labels and colors to active samples only
    active_samples = embeddings_df.sample_id.unique()
    labels_df = labels_df[labels_df['sample_id'].isin(active_samples)]
    populations = labels_df['population'].unique()
    colors_df = colors_df[colors_df['population'].isin(populations)]

    output_basename = args.output_name.removesuffix(".html")

    embeddings_df = embeddings_df[embeddings_df['time'] >= args.min_time]

    print("Building plot...")
    plotter = PlotEmbeddings(
        embeddings_df=embeddings_df,
        labels_df=labels_df,
        colors_df=colors_df,
        generation_time=args.generation_time,
    )
    plotter.init_plot(time_scale="log")
    plotter.add_sample_trajectories()

    tube_kwargs = dict(
        grid_res_xy=args.grid_res_xy,
        grid_res_u=args.grid_res_u,
        mass_frac=args.mass_frac,
        sigma_u=args.sigma_u,
        verbose=True,
    )

    print("Adding support tubes...")
    plotter.add_support_tube(**tube_kwargs)

    print("Adding support rings...")
    plotter.add_support_rings(**tube_kwargs)

    plotter.add_legend()

    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, f"{output_basename}.html")
    plotter.fig.write_html(out_file)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
