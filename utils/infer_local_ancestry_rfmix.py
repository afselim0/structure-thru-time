#%%
import argparse
import pandas as pd
import demes
import msprime
import numpy as np
import os
from tqdm import tqdm
from time import time
from pathos.multiprocessing import Pool

def main():
    parser = argparse.ArgumentParser(description="Simulate tree sequences from a demes file")
    parser.add_argument("--input_path", required=True, help="Path to tree sequence files")
    parser.add_argument("--output_path", required=True, help="Output directory for LAI dataframes")
    parser.add_argument("--pop_labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--r", type=float, default=1.2e-8, help="recombination_rate")
    parser.add_argument("--mu", type=float, default=1.2e-8, help="mutation_rate")
    parser.add_argument("--admix_pops", nargs='+',default=None, help="population labels to compute local ancestry for")
    parser.add_argument("--reference_pops", nargs='+',default=None, help="population labels to use as reference populations for local ancestry inference")
    parser.add_argument("--num_threads", type=int, default=1, help="number of threads to use")
   
    
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    pop_labels = pd.read_parquet(args.pop_labels)
    recomb_rate = args.r
    mut_rate = args.mu
    admix_pops = args.admix_pops
    reference_pops = args.reference_pops
    num_threads = int(args.num_threads)

    admix_samples = pop_labels[pop_labels["population"].isin(admix_pops)]["sample_id"].values if admix_pops is not None else None
    reference_samples_dict = {pop_label: pop_labels[pop_labels["population"] == pop_label]["sample_id"].values for pop_label in reference_pops} if reference_pops is not None else None


if __name__ == "__main__":
    main()