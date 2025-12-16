import os
import argparse
from utils import plot_demography
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Preprocess demography from a demes file")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file", type=str)
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for preprocessed demography data", type=str)
    parser.add_argument("--max_time", required=True, help="Maximum time for demography plot", type=int)

    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    colors = args.colors
    output_path = args.output_path
    max_time = args.max_time
    pop_colors = pd.read_parquet(colors)
    demes_fig = plot_demography(demes_file_path, pop_colors, max_time)

    demes_fig.savefig(os.path.join(output_path, "demography.png"), dpi=500)

if __name__ == "__main__":
    main()