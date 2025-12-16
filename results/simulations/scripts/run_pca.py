import argparse
from utils import run_pca, plot_pca
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run PCA on simulated genetic data")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for PCA results", type=str)
    parser.add_argument("--colors", required=True, help="path to colors file", type=str)
    parser.add_argument("--labels", required=True, help="path to labels file", type=str)
    parser.add_argument("--n", required=True, help="Number of samples per population", type=int)

    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    output_path = args.output_path
    n = args.n
    colors_path = args.colors
    labels_path = args.labels
    colors = pd.read_parquet(colors_path)
    labels = pd.read_parquet(labels_path)

    sequence_length = 1e7
    recombination_rate = 1e-8
    num_components = 2

    pca_df = run_pca(demes_file_path, n, sequence_length, recombination_rate, num_components)
    pca_fig = plot_pca(pca_df,labels,colors)

    pca_fig.savefig(os.path.join(output_path, "pca.png"), dpi=500)
    # pca_df.to_parquet(os.path.join(output_path, "pca_results.parquet"))

if __name__ == "__main__":
    main()