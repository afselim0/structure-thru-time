#%%
#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append("../src")
sys.path.append("../../src")
from compute_coal_rates import compute_coal_rates, compute_coal_rates_bootstrap_se
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--pdf_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for pdf data")
    parser.add_argument("--output_name", required=True, help="Output filename for pdf parquet file")
    parser.add_argument("--minimum_window_size", type=float, required=True, help="minimum window size for rate calculation")
    parser.add_argument("--maximum_window_size", type=float, required=True, help="maximum window size for rate calculation")
    parser.add_argument("--n_bootstrap_replicates", type=int, default=100, help="number of bootstrap replicates for SE calculation")
    parser.add_argument("--bootstrap_subsample_size", type=int, required=False, help="number of count files sampled per bootstrap replicate (without replacement)")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_path = args.output_path
    output_name = args.output_name
    minimum_window_size = args.minimum_window_size
    maximum_window_size = args.maximum_window_size
    n_bootstrap_replicates = args.n_bootstrap_replicates
    bootstrap_subsample_size = args.bootstrap_subsample_size

    print(f"computing rates from estimates of the pdfs of pairwise coalescent times: {args.pdf_path}")
    pdf_df = pd.read_parquet(pdf_path)
    pdf_df.columns = pdf_df.columns.astype(float)
    bootstrap_dir = output_path + "/bootstrap_pdf_samples"
    rates_df = compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size)
    rates_df.columns = rates_df.columns.astype(str)
    rates_df.to_parquet(f"{output_path}/{output_name}")

    if os.path.exists(bootstrap_dir) and len(os.listdir(bootstrap_dir)) > 0:
        n_files = len([f for f in os.listdir(bootstrap_dir) if f.endswith(".parquet") and f != "coverage.parquet"])
        subsample_size = bootstrap_subsample_size if bootstrap_subsample_size is not None else n_files // 2
        se_df = compute_coal_rates_bootstrap_se(
            pdf_df, bootstrap_dir,
            minimum_window_size, maximum_window_size,
            n_replicates=n_bootstrap_replicates,
            subsample_size=subsample_size,
        )
        se_df.columns = se_df.columns.astype(str)
        se_df.to_parquet(f"{output_path}/se_{output_name}")

if __name__ == "__main__":
    main()