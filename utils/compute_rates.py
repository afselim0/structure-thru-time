#%%
#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append("../src")
sys.path.append("../../src")
from ComputeCoalescentRates import compute_coal_rates, compute_coal_rates_variance
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--pdf_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for pdf data")
    parser.add_argument("--output_name", required=True, help="Output filename for pdf parquet file")
    parser.add_argument("--minimum_window_size", type=float, required=True, help="minimum window size for rate calculation")
    parser.add_argument("--maximum_window_size", type=float, required=True, help="maximum window size for rate calculation")
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true", help="If set, per-sample-pair (haplotype) rates are returned. Otherwise rates are averaged over pairs of haploid samples from the same diploid individual to return per-individual-pair rates.")
    parser.add_argument("--pop_labels", type=str, default=None, help="Path to population labels csv (with sample_id and individual_id columns). Used to map haploid samples to diploid individuals when phased=False. If omitted, consecutive sample IDs (2k, 2k+1) are assumed to belong to individual k.")
    parser.add_argument("--compute_variance", action="store_true", help="Compute standard errors using jackknife approach")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_path = args.output_path
    output_name = args.output_name
    minimum_window_size = args.minimum_window_size
    maximum_window_size = args.maximum_window_size
    compute_variance = args.compute_variance
    per_haplotype = args.per_haplotype
    print(f"computing rates from estimates of the pdfs of pairwise coalescent times: {args.pdf_path}")
    pdf_df = pd.read_parquet(pdf_path)
    pdf_df.columns = pdf_df.columns.astype(float)
    coal_counts_dir = output_path + "/counts_dir" + "/" + os.path.basename(pdf_path).replace(".parquet", "")

    # Sample-to-individual mapping for averaging haplotype pairs into individuals
    # (only used when phased=False). Passing it explicitly avoids relying on the
    # consecutive sample-ID fallback, which is invalid for subsampled empirical data.
    sample_individal_ids = None
    if not per_haplotype and args.pop_labels is not None:
        sample_individal_ids = pd.read_csv(args.pop_labels)[["sample_id", "individual_id"]]

    rates_df = compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size, per_haplotype=per_haplotype, sample_individal_ids=sample_individal_ids)
    rates_df.columns = rates_df.columns.astype(str)
    rates_df.to_parquet(f"{output_path}/{output_name}")

    if compute_variance:
        variance_df = compute_coal_rates_variance(
            pdf_df,
            coal_counts_dir,
            minimum_window_size,
            maximum_window_size,
            per_haplotype=per_haplotype,
            sample_individal_ids=sample_individal_ids
        )
        variance_df.columns = variance_df.columns.astype(str)
        variance_df.to_parquet(f"{output_path}/variance_{output_name}")

if __name__ == "__main__":
    main()