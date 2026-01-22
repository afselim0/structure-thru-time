#!/usr/bin/env python3

import argparse
import sys
import os
sys.path.append("../src")
sys.path.append("../../src")
from CountCoalEvents import CountCoalEvents
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--input_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for pdf data")
    parser.add_argument("--output_name", required=True, help="Output filename for pdf parquet file")
    parser.add_argument("--max_time", type=int, required=True, help="Maximum time for rate calculation")
    parser.add_argument("--n_timepoints", type=int, required=True, help="Number of timepoints for rate calculation")
    parser.add_argument("--num_posterior_samples", type=int, default=1, help="Number of posterior samples to use")
    parser.add_argument("--fraction_pairs", type=float, default=1, help="Fraction of pairs to sample for coalescence counting")
    parser.add_argument("--pops", nargs='+', help="Population labels to compute rates for")
    parser.add_argument("--n_samples_per_pop", type=int, default= -1, help="Number of samples to draw per population")
    parser.add_argument("--pop_labels", type=str, help="Path to population labels file")
    
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    max_time = args.max_time
    n_timepoints = args.n_timepoints
    num_posterior_samples = args.num_posterior_samples
    fraction_pairs = args.fraction_pairs
    chosen_pops = args.pops
    n_samples_per_pop = args.n_samples_per_pop
    pop_labels_path = args.pop_labels
    population_labels = pd.read_csv(pop_labels_path)
    
    chosen_pops = None if chosen_pops == "all" else chosen_pops
    n_samples_per_pop = None if n_samples_per_pop == -1 else n_samples_per_pop

    os.makedirs(output_path, exist_ok=True)
    
    if chosen_pops is None:
        if "sample_time" in population_labels.columns:
            population_labels = population_labels[population_labels["sample_time"]==0]
    else:
        population_labels = population_labels[population_labels["population"].isin(chosen_pops)]
    
    if n_samples_per_pop is not None:
        chosen_samples = []
        for pop in chosen_pops:
            sample_ids = population_labels[population_labels["population"] == pop]["sample_id"].values
            if n_samples_per_pop < len(sample_ids):
                subsample_ids = sample_ids[:n_samples_per_pop]
            else:
                subsample_ids = sample_ids
            chosen_samples.extend(subsample_ids)
    else:
        chosen_samples = None
    data_dir = os.path.join(output_path, "pdf_data")
    os.makedirs(data_dir, exist_ok=True)

    CCE = CountCoalEvents(ts_dir=input_path, num_posterior_samples=num_posterior_samples,max_time=max_time, num_timepoints=n_timepoints,fraction_pairs=fraction_pairs, sample_ids=chosen_samples)
    pdf_df = CCE.compute_coal_time_pdf(num_threads=10, data_output=data_dir)
    pdf_df.to_parquet(f"{output_path}/{args.output_name}")

if __name__ == "__main__":
    main()