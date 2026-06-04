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
    parser.add_argument("--phased", type=str, default="False", help="Indicates whether the input data is phased.  If False, rates are averaged over pairs of haploid samples from the same diploid individual to return per-individual-pair rates.  If True, per-sample-pair rates are returned.")
    parser.add_argument("--pop_labels", type=str, default=None, help="Path to population labels csv (with sample_id and individual_id columns). Used to map haploid samples to diploid individuals when phased=False. If omitted, consecutive sample IDs (2k, 2k+1) are assumed to belong to individual k.")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_path = args.output_path
    output_name = args.output_name
    minimum_window_size = args.minimum_window_size
    maximum_window_size = args.maximum_window_size
    n_bootstrap_replicates = args.n_bootstrap_replicates
    bootstrap_subsample_size = args.bootstrap_subsample_size
    phased = args.phased == "True"
    print(f"computing rates from estimates of the pdfs of pairwise coalescent times: {args.pdf_path}")
    pdf_df = pd.read_parquet(pdf_path)
    pdf_df.columns = pdf_df.columns.astype(float)
    bootstrap_dir = output_path + "/bootstrap_pdf_samples"

    # Sample-to-individual mapping for averaging haplotype pairs into individuals
    # (only used when phased=False). Passing it explicitly avoids relying on the
    # consecutive sample-ID fallback, which is invalid for subsampled empirical data.
    sample_individal_ids = None
    if not phased and args.pop_labels is not None:
        sample_individal_ids = pd.read_csv(args.pop_labels)[["sample_id", "individual_id"]]

    rates_df = compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size, phased=phased, sample_individal_ids=sample_individal_ids)
    rates_df.columns = rates_df.columns.astype(str)
    rates_df.to_parquet(f"{output_path}/{output_name}")

    # if os.path.exists(bootstrap_dir) and len(os.listdir(bootstrap_dir)) > 0:
    #     n_files = len([f for f in os.listdir(bootstrap_dir) if f.endswith(".parquet") and f != "coverage.parquet"])
    #     subsample_size = bootstrap_subsample_size if bootstrap_subsample_size is not None else n_files // 2
    #     se_df = compute_coal_rates_bootstrap_se(
    #         bootstrap_dir,
    #         minimum_window_size, maximum_window_size,
    #         n_replicates=n_bootstrap_replicates,
    #         subsample_size=subsample_size,
    #     )
    #     se_df.columns = se_df.columns.astype(str)
    #     se_df.to_parquet(f"{output_path}/se_{output_name}")

if __name__ == "__main__":
    main()