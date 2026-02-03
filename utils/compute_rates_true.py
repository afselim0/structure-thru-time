#!/usr/bin/env python3
import argparse
import pandas as pd
import demes
import msprime
import numpy as np


def compute_true_rates(demes_file_path, max_time, delta_t = 10):
    graph = demes.load(demes_file_path)
    demography = msprime.Demography.from_demes(graph)
    populations = np.array([p.name for p in demography.populations if p.default_sampling_time==0])
    demography_debugger = demography.debug()
    times = np.arange(0,max_time,delta_t)
    rates_df = pd.DataFrame(columns=times,index=pd.MultiIndex.from_product([[], []], names=["pop_1", "pop_2"]))
    pop_pairs = []
    for i in range(len(populations)):
        for j in range(i,len(populations)):
            pop1 = populations[i]
            pop2 = populations[j]
            pop_pairs.append((pop1,pop2))
    for pair in pop_pairs:
            pop1 = pair[0]
            pop2 = pair[1]
            if pop1 == pop2:
                samples = {pop1:2}
            else:
                samples = {pop1:1,pop2:1}
            coalescence_rates = demography_debugger.coalescence_rate_trajectory(steps=times,lineages=samples,double_step_validation=False)[0]
            rates_df.loc[(pop1,pop2),:] = coalescence_rates
    return rates_df

def expand_true_rates(rates_df, pop_labels):
    sample_pop_dict = dict(zip(pop_labels['sample_id'], pop_labels['population']))
    samples = pop_labels['sample_id'].tolist()
    sample_pairs = []
    for i in range(len(samples)):
        for j in range(i+1,len(samples)):
            sample1 = samples[i]
            sample2 = samples[j]
            sample_pairs.append((sample1,sample2))
    sample_rates_df = pd.DataFrame(columns=rates_df.columns,index=pd.MultiIndex.from_tuples(sample_pairs,names=["sample_1","sample_2"]))
    for pair in sample_pairs:
        sample1 = pair[0]
        sample2 = pair[1]
        pop1 = sample_pop_dict[sample1]
        pop2 = sample_pop_dict[sample2]
        try:
            rates = rates_df.loc[(pop1,pop2),:]
        except:
            rates = rates_df.loc[(pop2,pop1),:]
        sample_rates_df.loc[(sample1,sample2),:] = rates
    return sample_rates_df


def main():
    parser = argparse.ArgumentParser(description="Compute true coalescence rates from a demes file")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file")
    parser.add_argument("--output_path", required=True, help="Output directory for rates data")
    parser.add_argument("--max_time", type=int, required=True, help="Maximum time for rate calculation")
    parser.add_argument("--labels", type=str, required=True, help="Path to the population labels file")
    parser.add_argument("--delta_t", type=int, default=10, help="Time step for rate calculation")
    
    args = parser.parse_args()
    
    demes_file_path = args.demes_file_path
    output_path = args.output_path
    max_time = args.max_time
    labels_path = args.labels
    delta_t = args.delta_t

    # if delta_t is None:
    #     delta_t = max_time // 1000  # Default to 1000 time points

    pop_rates_df = compute_true_rates(demes_file_path, max_time, delta_t=delta_t)
    pop_labels = pd.read_csv(labels_path)
    expanded_rates_df = expand_true_rates(pop_rates_df, pop_labels)
    expanded_rates_df.columns = expanded_rates_df.columns.astype(str)
    expanded_rates_df.to_parquet(f"{output_path}/rates_true.parquet")

if __name__ == "__main__":
    main()
