import argparse
import pandas as pd
import demes
import msprime
import numpy as np
import os
from tqdm import tqdm
from time import time
from pathos.multiprocessing import Pool

def sim_contigs(demes_file_path, output_dir, pop_labels, num_contigs=500, contig_length=5e6, num_replicates=1, recomb_rate=1e-8, parallel=True):
    graph = demes.load(demes_file_path)
    demography = msprime.Demography.from_demes(graph)
    populations = np.unique(pop_labels['population'])
    n_samples = len(pop_labels)
    n_pops = len(populations)
    n_samples_per_pop = n_samples // n_pops
    samples = {pop_labels.loc[i*n_samples_per_pop,'population']: n_samples_per_pop for i in range(n_pops)}


    def simulate_replicate(i):
        ts = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            sequence_length=contig_length,
            recombination_rate=recomb_rate,
            ploidy=1,
            random_seed=42+i
        )
        return i, ts
    
    start_time = time()
    if parallel:
        with Pool(os.cpu_count()) as pool:
            for i, ts in tqdm(pool.imap(simulate_replicate, range(num_contigs*num_replicates)), total=num_contigs*num_replicates):
                if num_replicates == 1:
                    ts.dump(f"{output_dir}/contig_{i}.trees")
                else:
                    rep = i // num_contigs
                    contig = i % num_contigs
                    ts.dump(f"{output_dir}/contig_{contig}_{rep}.trees")
    else:
        for i in tqdm(range(num_contigs*num_replicates)):
            _, ts = simulate_replicate(i)
            if num_replicates == 1:
                ts.dump(f"{output_dir}/contig_{i}.trees")
            else:
                rep = i // num_contigs
                contig = i % num_contigs
                ts.dump(f"{output_dir}/contig_{contig}_{rep}.trees")
    end_time = time()
    time_taken = end_time - start_time
    
    return time_taken

def main():
    parser = argparse.ArgumentParser(description="Simulate tree sequences from a demes file")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file")
    parser.add_argument("--output_path", required=True, help="Output directory for simulated trees")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    
    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    output_path = args.output_path
    labels = pd.read_parquet(args.labels)
    
    time_taken = sim_contigs(demes_file_path, output_path, labels)
    with open(f"{output_path}/finished_sim.txt", "w") as f:
        f.write(f"Simulation took {time_taken/60:.2f} minutes")

if __name__ == "__main__":
    main()
