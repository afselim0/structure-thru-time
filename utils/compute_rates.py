#!/usr/bin/env python3
import argparse 
import sys
sys.path.append("../src")
sys.path.append("../../src")
from compute_coal_rates import compute_coal_rates
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--pdf_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for pdf data")
    parser.add_argument("--output_name", required=True, help="Output filename for pdf parquet file")
    parser.add_argument("--minimum_window_size", type=float, required=True, help="minimum window size for rate calculation")
    parser.add_argument("--maximum_window_size", type=float, required=True, help="maximum window size for rate calculation")
    parser.add_argument("--poly", type=int, default=2, help="degree of local polynomial fit for rate calculation")
    args = parser.parse_args()
    
    print(f"computing rates from estimates of the pdfs of pairwise coalescent times: {args.pdf_path}")
    pdf_df = pd.read_parquet(args.pdf_path)
    pdf_df.columns = pdf_df.columns.astype(float)
    rates_df,se_df = compute_coal_rates(pdf_df, args.minimum_window_size, args.maximum_window_size)
    rates_df.columns = rates_df.columns.astype(str)
    se_df.columns = se_df.columns.astype(str)
    rates_df.to_parquet(f"{args.output_path}/{args.output_name}")
    se_df.to_parquet(f"{args.output_path}/se_{args.output_name}")
if __name__ == "__main__":
    main()