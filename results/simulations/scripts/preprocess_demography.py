import os
import argparse
from utils import preprocess_demography

def main():
    parser = argparse.ArgumentParser(description="Preprocess demography from a demes file")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for preprocessed demography data", type=str)
    parser.add_argument("--n", required=True, help="Number of samples per population", type=int)

    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    output_path = args.output_path
    n = args.n
    pop_labels, pop_colors = preprocess_demography(demes_file_path=demes_file_path,n_samples_per_pop=n)
    pop_colors.to_parquet(os.path.join(output_path,"pop_colors.parquet"))
    pop_labels.to_parquet(os.path.join(output_path,"pop_labels.parquet"))

if __name__ == "__main__":
    main()