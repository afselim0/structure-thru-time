import argparse
import sys
import pandas as pd
sys.path.append("../../../src")
from PlotEmbeddings import PlotEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Plot embeddings")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    
    args = parser.parse_args()
    embeddings_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    output_path = args.output_path


    active_samples = embeddings_df.sample_id.unique()
    labels_df = labels_df[labels_df['sample_id'].isin(active_samples)]
    populations = labels_df['population'].unique()
    colors_df = colors_df[colors_df['population'].isin(populations)]


    plotter = PlotEmbeddings(embeddings_df,labels_df,colors_df)
    embeddings_dyn_fig = plotter.dynamic_plot(log_time=True)
    embeddings_dyn_fig.write_html(f"{output_path}/embeddings.html")
    embeddings_dyn_fig.show()

    # # new code needs debugging
    # embeddings_stat_fig = plotter.dynamic_plot(log_time=True)
    # embeddings_stat_fig.write_image(f"{output_path}/embeddings.png",scale=2)
    
    
    
if __name__ == "__main__":
    main()
