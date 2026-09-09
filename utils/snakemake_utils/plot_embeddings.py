#%%
#!/usr/bin/env python3
import argparse
import sys
import os
import pandas as pd
from plot_embeddings_mp4 import *
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
# import support_tube

from PlotEmbeddings import PlotEmbeddings


STATIC_EXPORT_HINT = (
    "static PNG export failed; the interactive HTML is unaffected. plotly's kaleido needs a "
    "Chrome binary. One-time fix per user:\n"
    "  SSL_CERT_FILE=$(python -c \"import certifi;print(certifi.where())\") "
    "python -c \"from kaleido import get_chrome_sync; print(get_chrome_sync())\""
)


def save_static_png(plotter, path, **kwargs):
    """Render one static snapshot to path; on failure print a warning and return False."""
    try:
        img, _ = plotter.save_static(**kwargs)
    except Exception as e:  # ChromeNotFoundError surfaces as RuntimeError from plotly
        print(f"WARNING: {path}: {STATIC_EXPORT_HINT}\n  ({type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ''})",
              file=sys.stderr)
        return False
    img.save(path)
    return True


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
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true", help="If set, trajectories are per-haplotype (sample_id). Otherwise per-individual (individual_id)")
    parser.add_argument("--param_label", default=None, type=str, help="Optional parameter string rendered as plot subtitle")

    args = parser.parse_args()
    per_haplotype = args.per_haplotype
    id_label = "sample_id" if per_haplotype else "individual_id"
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
    if not per_haplotype and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    active_samples = embeddings_df[id_label].unique()
    labels_df = labels_df[labels_df[id_label].isin(active_samples)]
    # When unphased, each individual_id appears once per haplotype; collapse to one
    # row per id so the merge in PlotEmbeddings does not duplicate trajectories.
    labels_df = labels_df[[id_label, "population"]].drop_duplicates(subset=id_label)
    populations = labels_df['population'].unique()
    colors_df = colors_df[colors_df['population'].isin(populations)]
    # If ancient samples are detected (samples where the embedding is NaN), print who they are
    ancient_samples = embeddings_df[embeddings_df.isna().any(axis=1)]
    if not ancient_samples.empty:
        print("Ancient samples detected:")
        # print unique sample ids
        print(ancient_samples.index.get_level_values(0).unique())
    else:
        print("No ancient samples detected.")

    output_basename = output_name.removesuffix(".html")
    print(f"Building Embeddings plots for embeddings file: {args.input}")

    embeddings_df = embeddings_df[embeddings_df['time'] >= min_time]
    plotter = PlotEmbeddings(embeddings_df=embeddings_df, labels_df=labels_df, colors_df=colors_df, sample_id_col=id_label)#, generation_time=generation_time)
    plotter.init_plot(time_scale=args.time_scale, xy_lims=lims, min_time=min_time, max_time=max_time)
    plotter.add_sample_trajectories()
    # plotter.add_population_medians()
    plotter.add_legend()
    title_text = (f"{output_basename}<br><sup>{args.param_label}</sup>"
                  if args.param_label else output_basename)
    plotter.fig.update_layout(title_text=title_text)
    plotter.fig.write_html(f"{output_path}/{output_basename}.html")
    # The PNG snapshots are a convenience, not a pipeline output: if static export
    # is unavailable (kaleido needs a Chrome binary, which midway3 compute nodes
    # lack) warn and carry on rather than failing the whole run.
    save_static_png(plotter, f"{output_path}/{output_basename}_v1.png",
                    width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1))
    save_static_png(plotter, f"{output_path}/{output_basename}_v2.png",
                    width=600, height=600, scale=2, eye=dict(x=-0.75, y=-1.5, z=0.3), up=dict(x=0, y=0, z=1))

    try:
        smooth_embeddings = pd.read_parquet(f"{args.input.replace('embeddings', 'smooth_embeddings')}")
        print(f"Building Smoothed Embeddings plots for embeddings file: {args.input.replace('embeddings', 'smooth_embeddings')}")
        plotter_smooth = PlotEmbeddings(embeddings_df=smooth_embeddings, labels_df=labels_df, colors_df=colors_df, sample_id_col=id_label)#, generation_time=generation_time)
        plotter_smooth.init_plot(time_scale=args.time_scale, xy_lims=lims, min_time=min_time, max_time=max_time)
        plotter_smooth.add_sample_trajectories()
        # plotter_smooth.add_population_medians()
        plotter_smooth.add_legend()
        smooth_title = (f"{output_basename} (smoothed)<br><sup>{args.param_label}</sup>"
                        if args.param_label else output_basename + " (smoothed)")
        plotter_smooth.fig.update_layout(title_text=smooth_title)
        plotter_smooth.fig.write_html(f"{output_path}/smooth_{output_basename}.html")
    except FileNotFoundError:
        print("Smooth embeddings file not found.")
        
    # if len(populations) <=5:
    #     plotter.add_support_tube()
    #     plotter.fig.write_html(f"{output_path}/convexhull_{output_basename}.html")

    #     plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = labels_df,colors_df = colors_df)#, generation_time = generation_time)
    #     plotter.init_plot(time_scale=args.time_scale,xy_lims=lims, min_time=min_time, max_time=max_time)
    #     plotter.add_support_tube()
    #     plotter.fig.write_html(f"{output_path}/convexhullONLY_{output_basename}.html")
    
    # print("Creating MP4 animation...")
    # try:
    #     make_mp4(
    #         parquet_path = args.input,
    #         out_mp4 = f"{output_path}/{output_name.rstrip('.html')}_animation",
    #         fps = 20,
    #         bounce = True,
    #         point_size = 10.0,
    #         alpha = 0.8,
    #         dpi = 120,
    #         min_time = args.min_time,
    #         max_time = args.max_time,
    #         colors_path = args.colors,
    #         labels_path = args.labels,
    #     )
    # except Exception as e:
    #     print(f"Warning: MP4 animation failed: {e}")

if __name__ == "__main__":
    main()
