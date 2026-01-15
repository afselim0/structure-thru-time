import argparse
import sys
sys.path.append("../src")
from CountCoalEvents import compute_coal_rates
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--pdf_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for pdf data")
    parser.add_argument("--output_name", required=True, help="Output filename for pdf parquet file")
    parser.add_argument("--window_size", type=float, required=True, help="window size for rate calculation")
    args = parser.parse_args()
    
    pdf_df = pd.read_parquet(args.pdf_path)
    rates_df = compute_coal_rates(pdf_df, args.window_size)
    rates_df.to_parquet(f"{args.output_path}/{args.output_name}")
if __name__ == "__main__":
    main()