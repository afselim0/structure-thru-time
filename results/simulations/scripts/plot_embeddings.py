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
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    
    args = parser.parse_args()
    embeddings_df = pd.read_parquet(args.input)
    colors_df = pd.read_parquet(args.colors)
    labels_df = pd.read_parquet(args.labels)
    output_path = args.output_path
    output_name = args.output_name
    plotter = PlotEmbeddings(embeddings_df,labels_df,colors_df)
    embeddings_fig = plotter.dynamic_plot(log_time=True)
    embeddings_fig.write_html(f"{output_path}/{output_name}")
    embeddings_fig.show()
if __name__ == "__main__":
    main()
