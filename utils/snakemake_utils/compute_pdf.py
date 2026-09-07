#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append("../src")
sys.path.append("../../src")
from CountCoalEvents import CountCoalEvents, CountCoalEvents_admix
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--input_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for pdf data")
    parser.add_argument("--output_name", required=True, help="Output filename for pdf parquet file")
    parser.add_argument("--max_time", type=int, required=True, help="Maximum time for rate calculation")
    parser.add_argument("--n_timepoints", type=int, required=True, help="Number of timepoints for rate calculation")
    parser.add_argument("--fraction_samples", type=float, default=1, help="Fraction of samples to use for coalescence counting")
    parser.add_argument("--pop_labels", type=str, help="Path to population labels file (already sliced to the samples used in this run)")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to use for computation")
    parser.add_argument("--LAI_dir", type=str, default=None, help="Directory of per-contig local-ancestry parquet files. If set together with --ancestry_STT_dir, also compute per-ancestry coalescent-time PDFs.")
    parser.add_argument("--ancestry_STT_dir", type=str, default=None, help="Directory with one subdirectory per ancestry label, each containing data/pop_labels.csv. If set together with --LAI_dir, also compute per-ancestry coalescent-time PDFs.")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    max_time = args.max_time
    n_timepoints = args.n_timepoints
    num_threads = args.num_threads
    fraction_samples = args.fraction_samples
    pop_labels_path = args.pop_labels
    LAI_dir = args.LAI_dir
    ancestry_STT_dir = args.ancestry_STT_dir
    population_labels = pd.read_csv(pop_labels_path)
    chosen_samples = population_labels["sample_id"].values

    if (LAI_dir is None) != (ancestry_STT_dir is None):
        raise ValueError("--LAI_dir and --ancestry_STT_dir must be provided together to enable admix mode.")
    admix_mode = LAI_dir is not None and ancestry_STT_dir is not None

    print(f"estimating the probability density funciton of pairwise coalescent times for trees in {input_path}")
    os.makedirs(output_path, exist_ok=True)

    coal_counts_dir = os.path.join(output_path, "counts_dir")
    coal_counts_dir = os.path.join(coal_counts_dir, f"{args.output_name.removesuffix('.parquet')}")
    os.makedirs(coal_counts_dir, exist_ok=True)
    CCE = CountCoalEvents(ts_dir=input_path ,max_time=max_time, num_timepoints=n_timepoints,fraction_samples=fraction_samples, sample_ids=chosen_samples)
    CCE.count_coal_spans_all_ts(coal_counts_dir=coal_counts_dir, num_threads=num_threads)
    pdf_df = CCE.compute_coal_time_pdf(coal_counts_dir=coal_counts_dir)
    pdf_df.columns = pdf_df.columns.astype(str)
    pdf_df.to_parquet(f"{output_path}/{args.output_name}")

    if admix_mode:
        print(f"admix mode: computing per-ancestry coalescent-time PDFs using LAI data in {LAI_dir}")
        ancestry_labels = sorted(
            d for d in os.listdir(ancestry_STT_dir) if os.path.isdir(os.path.join(ancestry_STT_dir, d))
        )
        if not ancestry_labels:
            raise ValueError(f"No ancestry subdirectories found in ancestry_STT_dir: {ancestry_STT_dir}")

        pop_labels_paths = {
            label: os.path.join(ancestry_STT_dir, label, "data", "pop_labels.csv") for label in ancestry_labels
        }
        ancestry_count_dirs = {
            label: os.path.join(ancestry_STT_dir, label, "data", "counts_dir", args.output_name.removesuffix('.parquet'))
            for label in ancestry_labels
        }

        CCE_admix = CountCoalEvents_admix(
            cce=CCE,
            coal_counts_dir=coal_counts_dir,
            LAI_dir=LAI_dir,
            ancestry_count_dirs=ancestry_count_dirs,
            pop_labels_paths=pop_labels_paths,
        )
        CCE_admix.count_coal_spans_all_ts(LAI_dir=LAI_dir, num_threads=num_threads)

        for label in ancestry_labels:
            ancestry_pdf_df = CCE_admix.compute_coal_time_pdf(label)
            ancestry_pdf_df.columns = ancestry_pdf_df.columns.astype(str)
            ancestry_pdf_path = os.path.join(ancestry_STT_dir, label, "data", args.output_name)
            ancestry_pdf_df.to_parquet(ancestry_pdf_path)
            print(f"saved ancestry '{label}' pdf to {ancestry_pdf_path}")

if __name__ == "__main__":
    main()