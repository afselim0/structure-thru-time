#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
sys.path.append("../../src")
sys.path.append("../src")
sys.path.append("src")

from PlotEmbeddings import PlotEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Plot embeddings")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    parser.add_argument("--min_time", default=0, type=float,help="Minimum time for plotting")
    parser.add_argument("--lims", type=str, help="Minimum time for plotting")
    parser.add_argument("--generation_time", default=None, type=float,help="Generation time in years")

    args = parser.parse_args()
    embeddings_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    output_path = args.output_path
    output_name = args.output_name
    min_time = args.min_time
    generation_time = args.generation_time
    if args.lims == "None":
        lims = None
    else:
        lims = float(args.lims)
    
    # subsetting labels_df and colors_df to active samples only
    active_samples = embeddings_df.sample_id.unique()
    labels_df = labels_df[labels_df['sample_id'].isin(active_samples)]
    populations = labels_df['population'].unique()
    colors_df = colors_df[colors_df['population'].isin(populations)]

    output_basename = output_name.removesuffix(".html")
    print(f"Building Embeddings plots for embeddings file: {args.input}")

    embeddings_df = embeddings_df[embeddings_df['time'] >= min_time]
    plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = labels_df,colors_df = colors_df)#, generation_time = generation_time)
    plotter.init_plot(time_scale="log",xy_lims=args.lims)
    plotter.add_sample_trajectories()
    # plotter.add_population_medians()
    plotter.add_legend()
    # plotter.save_static(filename=f"{output_path}/{output_basename}_v1", format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1))
    # plotter.save_static(filename=f"{output_path}/{output_basename}_v2", format="png", width=600, height=600, scale=2, eye=dict(x=-0.75, y=-1.5, z=0.3), up=dict(x=0, y=0, z=1))
    plotter.fig.write_html(f"{output_path}/{output_basename}.html")
    # plotter.add_convex_hull()
    # plotter.fig.write_html(f"{output_path}/convexhull_{output_basename}.html")
    plotter.fig.show()
    
if __name__ == "__main__":
    main()
