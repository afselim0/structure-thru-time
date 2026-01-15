import argparse
import os
import sys
sys.path.append("../../../src/")
from CountCoalEvents import CountCoalEvents,compute_coal_rates
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--input_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for rates data")
    parser.add_argument("--output_name", required=True, help="Output filename for rates parquet file")
    parser.add_argument("--max_time", type=int, required=True, help="Maximum time for rate calculation")
    parser.add_argument("--window_size", type=int, required=True, help="Window size for rate calculation")
    parser.add_argument("--n_timepoints", type=int, required=True, help="Number of timepoints for rate calculation")
    parser.add_argument("--num_posterior_samples", type=int, required=True, help="Number of posterior samples to use")
    parser.add_argument("--fraction_pairs", type=float, default=1, help="Fraction of pairs to sample for coalescence counting")
    
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    output_name = args.output_name
    max_time = args.max_time
    window_size = args.window_size
    n_timepoints = args.n_timepoints
    num_posterior_samples = args.num_posterior_samples
    fraction_pairs = args.fraction_pairs
    # time_points = np.logspace(0, np.log10(max_time), n_timepoints).astype(int)
    # window_size = np.array([window_size]*n_timepoints)

    data_dir_name = f"{output_name.split('.')[0][6:]}_survival_data"
    data_dir = os.path.join(output_path, data_dir_name)
    os.makedirs(data_dir, exist_ok=True)
    CCE = CountCoalEvents(ts_dir=input_path, num_posterior_samples=num_posterior_samples,max_time=max_time, num_timepoints=n_timepoints,fraction_pairs=fraction_pairs)
    pdf_df = CCE.compute_coal_time_pdf(num_threads=os.cpu_count()-1, data_output=f"counts_data_{os.path.basename(output_name)}")
    rates_df = compute_coal_rates(pdf_df=pdf_df, window_size=window_size)
    rates_df.to_parquet(f"{output_path}/{output_name}")
    # pdf_df.to_parquet(f"{data_dir}/pdf_{output_name}")

if __name__ == "__main__":
    main()