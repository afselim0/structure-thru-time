#%%
#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
from plot_embeddings_mp4 import *
sys.path.append("../../src")
sys.path.append("../src")
sys.path.append("src")
# import support_tube

from PlotEmbeddings import PlotEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Plot embeddings")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    parser.add_argument("--min_time", type=float,help="Minimum time for plotting")
    parser.add_argument("--max_time", type=float,help="Maximum time for plotting")
    parser.add_argument("--lims", type=str, help="Minimum time for plotting")
    parser.add_argument("--generation_time", default=None, type=float,help="Generation time in years")
    parser.add_argument("--time_scale", default="linear", help="Time scale for demography plot, linear or log", type=str)

    args = parser.parse_args()
    embeddings_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    output_path = args.output_path
    output_name = args.output_name
    min_time = args.min_time
    max_time = args.max_time
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

    # embeddings_df = embeddings_df[embeddings_df['time'] >= min_time]
    plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = labels_df,colors_df = colors_df)#, generation_time = generation_time)
    plotter.init_plot(time_scale=args.time_scale,xy_lims=lims, min_time=min_time, max_time=max_time)
    plotter.add_sample_trajectories()
    # plotter.add_population_medians()
    plotter.add_legend()
    plotter.fig.show()
    plotter.save_static(filename=f"{output_path}/{output_basename}_v1", format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1))
    plotter.save_static(filename=f"{output_path}/{output_basename}_v2", format="png", width=600, height=600, scale=2, eye=dict(x=-0.75, y=-1.5, z=0.3), up=dict(x=0, y=0, z=1))
    plotter.fig.update_layout(title_text=output_basename)
    plotter.fig.write_html(f"{output_path}/{output_basename}.html")
    if len(populations) <=5:
        plotter.add_support_tube()
        plotter.fig.write_html(f"{output_path}/convexhull_{output_basename}.html")

        plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = labels_df,colors_df = colors_df)#, generation_time = generation_time)
        plotter.init_plot(time_scale=args.time_scale,xy_lims=lims, min_time=min_time, max_time=max_time)
        plotter.add_support_tube()
        plotter.fig.write_html(f"{output_path}/convexhullONLY_{output_basename}.html")
    
    print("Creating MP4 animation...")
    make_mp4(
    parquet_path = args.input,
    out_mp4 = f"{output_path}/{output_name.rstrip('.html')}_animation",
    fps = 20,
    bounce = True,
    point_size = 10.0,
    alpha = 0.8,
    dpi = 120,
    min_time = args.min_time,
    max_time = args.max_time,
    colors_path = args.colors,
    labels_path = args.labels,
    )

if __name__ == "__main__":
    main()
