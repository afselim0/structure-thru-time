import argparse
import pandas as pd
import demes
import msprime
import numpy as np
import os
from tqdm import tqdm
from time import time
from pathos.multiprocessing import Pool

def sim_contigs(demes_file_path, output_dir, pop_labels, num_contigs, contig_length, recomb_rate=1.2e-8, mutation_rate=1.2e-8, num_threads=1):
    graph = demes.load(demes_file_path)
    demography = msprime.Demography.from_demes(graph)
    populations = np.unique(pop_labels['population'])
    n_samples = len(pop_labels)
    n_pops = len(populations)
    n_samples_per_pop = n_samples // (2 * n_pops) # diploid samples
    samples = {pop_labels.loc[2*i*n_samples_per_pop,'population']: n_samples_per_pop for i in range(n_pops)} # diploid samples


    def simulate_replicate(i):
        ts = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            sequence_length=contig_length,
            recombination_rate=recomb_rate,
            ploidy=2,
            random_seed=42+i
        )
        return i, ts
    
    start_time = time()

    with Pool(num_threads) as pool:
        for i, ts in tqdm(pool.imap(simulate_replicate, range(num_contigs)), total=num_contigs):
            file_name = f"contig_{i}"
            ts.dump(f"{output_dir}/{file_name}.trees")
            ts = msprime.sim_mutations(ts,rate=mutation_rate,random_seed=42+i)
            with open(f"{output_dir}/{file_name}.vcf", "w") as f:
                ts.write_vcf(f, contig_id=str(i),allow_position_zero=True)

    end_time = time()
    time_taken = end_time - start_time
    
    return time_taken

def main():
    parser = argparse.ArgumentParser(description="Simulate tree sequences from a demes file")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file")
    parser.add_argument("--output_path", required=True, help="Output directory for simulated trees")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--num_contigs",type=float, required=True, help="number of contigs to simulate")
    parser.add_argument("--contig_length",type=float, required=True, help="length of each contig")
    parser.add_argument("--sim_params",type=str, required=True, help="simulation_parameters")   
    parser.add_argument("--mu", type=float, default=1.2e-8, help="mutation_rate")
    parser.add_argument("--r", type=float, default=1.2e-8, help="recombination_rate")
    parser.add_argument("--num_threads", type=int, default=1, help="number of threads to use")
   
    
    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    output_path = args.output_path
    labels = pd.read_csv(args.labels)
    num_contigs = int(args.num_contigs)
    contig_length = int(args.contig_length)
    sim_params = args.sim_params
    mu = args.mu
    r = args.r
    num_threads = int(args.num_threads)
    time_taken = sim_contigs(demes_file_path, output_path, labels, num_contigs, contig_length, recomb_rate=r, mutation_rate=mu, num_threads=num_threads)
    with open(f"{output_path}/{sim_params}", "w") as f:
        f.write(f"Simulation took {time_taken/60:.2f} minutes")

if __name__ == "__main__":
    main()

# %%
