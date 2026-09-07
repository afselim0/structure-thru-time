#!/usr/bin/env python3
import argparse
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from PlotEmbeddingsPyVista import PlotEmbeddingsPyVista


def main():
    parser = argparse.ArgumentParser(description="Plot embeddings (PyVista)")
    parser.add_argument("--input",       required=True,  help="Path to embeddings parquet file")
    parser.add_argument("--colors",      required=True,  help="Path to population colors CSV")
    parser.add_argument("--labels",      required=True,  help="Path to population labels CSV")
    parser.add_argument("--output_path", required=True,  help="Output directory")
    parser.add_argument("--output_name", required=True,  help="Output filename (HTML)")
    parser.add_argument("--min_time",    type=float,     help="Minimum time to display")
    parser.add_argument("--max_time",    type=float,     help="Maximum time to display")
    parser.add_argument("--lims",        type=str,       default="None", help="XY axis limit")
    parser.add_argument("--time_scale",  type=str,       default="linear", help="log or linear")
    parser.add_argument("--time_stride", type=int,       default=5,
                        help="Keep every Nth time point (reduces render cost)")
    args = parser.parse_args()

    embeddings_df = pd.read_parquet(args.input)
    colors_df     = pd.read_csv(args.colors)
    labels_df     = pd.read_csv(args.labels)
    lims          = None if args.lims == "None" else float(args.lims)
    output_basename = args.output_name.removesuffix(".html")

    # Restrict to active samples
    active_samples = embeddings_df.sample_id.unique()
    labels_df = labels_df[labels_df["sample_id"].isin(active_samples)]
    populations = labels_df["population"].unique()
    colors_df = colors_df[colors_df["population"].isin(populations)]

    print(f"Building PyVista embeddings plot for: {args.input}")

    plotter = PlotEmbeddingsPyVista(
        embeddings_df=embeddings_df,
        labels_df=labels_df,
        colors_df=colors_df,
    )
    plotter.init_plot(
        time_scale=args.time_scale,
        xy_lims=lims,
        min_time=args.min_time,
        max_time=args.max_time,
        time_stride=args.time_stride,
    )
    plotter.add_sample_trajectories()
    plotter.add_legend()
    plotter.show_axes()

    # Static PNGs
    plotter.save_static(f"{args.output_path}/{output_basename}_v1.png")

    # Interactive HTML via trame
    try:
        plotter.save_html(f"{args.output_path}/{output_basename}.html")
    except Exception as e:
        print(f"Warning: HTML export failed (trame may not be installed): {e}")


if __name__ == "__main__":
    main()
