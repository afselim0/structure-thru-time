#!/usr/bin/env python3
import argparse
import sys
import os
import numpy as np
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
    parser.add_argument("--fraction_samples", type=float, default=1, help="Fraction of samples to use for coalescence counting")
    parser.add_argument("--pops", nargs='+',default=None, help="Population labels to compute rates for")
    parser.add_argument("--n_samples_per_pop", type=int, default= -1, help="Number of samples to draw per population")
    parser.add_argument("--pop_labels", type=str, help="Path to population labels file")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of threads to use for computation")
    
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    max_time = args.max_time
    n_timepoints = args.n_timepoints
    num_posterior_samples = args.num_posterior_samples
    num_threads = args.num_threads
    fraction_samples = args.fraction_samples
    chosen_pops = args.pops
    n_samples_per_pop = args.n_samples_per_pop
    pop_labels_path = args.pop_labels
    population_labels = pd.read_csv(pop_labels_path)
    n_samples_per_pop = None if n_samples_per_pop == -1 else n_samples_per_pop
    chosen_pops = None if chosen_pops == ["all"] else chosen_pops

    print(f"estimating the probability density funciton of pairwise coalescent times for trees in {input_path}")
    os.makedirs(output_path, exist_ok=True)
    all_pops = population_labels["population"].unique()
    assert chosen_pops is None or np.all(np.isin(chosen_pops, all_pops)), "Some chosen populations are not in the population labels file."

    population_labels = population_labels[population_labels["population"].isin(chosen_pops)] if chosen_pops is not None else population_labels

    if n_samples_per_pop is None and chosen_pops is not None:
        chosen_samples = population_labels["sample_id"].values

    if n_samples_per_pop is not None:
        chosen_samples = []
        for pop in population_labels["population"].unique():
            sample_ids = population_labels[population_labels["population"] == pop]["sample_id"].values
            if n_samples_per_pop < len(sample_ids):
                subsample_ids = sample_ids[:n_samples_per_pop]
            else:
                subsample_ids = sample_ids
            chosen_samples.extend(subsample_ids)
    elif chosen_pops is not None:
        chosen_samples = population_labels["sample_id"].values
    elif chosen_pops is None and n_samples_per_pop is None:
        chosen_samples = None

    data_dir = os.path.join(output_path, "pdf_data")
    os.makedirs(data_dir, exist_ok=True)
    
    CCE = CountCoalEvents(ts_dir=input_path, num_posterior_samples=num_posterior_samples,max_time=max_time, num_timepoints=n_timepoints,fraction_samples=fraction_samples, sample_ids=chosen_samples)
    pdf_df = CCE.compute_coal_time_pdf(num_threads=num_threads, data_output=data_dir)
    pdf_df.columns = pdf_df.columns.astype(str)
    pdf_df.to_parquet(f"{output_path}/{args.output_name}")

if __name__ == "__main__":
    main()