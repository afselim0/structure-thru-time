#%%
#!/usr/bin/env python3
import argparse
import tskit
import msprime
import tsinfer
import tsdate
import glob
import os
# from pathos.multiprocessing import Pool
from multiprocessing import Pool

from time import time
from tqdm import tqdm

def infer_trees(sim_tree_path, infer_tree_path, mutation_rate=1e-8):
    ts_files = glob.glob(os.path.join(sim_tree_path, "*.trees"))
    os.makedirs(infer_tree_path, exist_ok=True)
    args = [
        (ts_file, infer_tree_path, mutation_rate)
        for ts_file in ts_files
    ]
    start_time = time()
    with Pool(os.cpu_count()) as pool:
        list(tqdm(pool.imap(process_file, args), total=len(args)))
    return time() - start_time

    
def process_file(args):
    ts_file, infer_tree_path, mutation_rate = args
    inferred_ts = infer(ts_file, mutation_rate=mutation_rate)
    base_name = os.path.basename(ts_file)
    inferred_ts.dump(os.path.join(infer_tree_path, base_name))

def infer(ts_file, mutation_rate=1e-8):
    ts = tskit.load(ts_file)
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=42)
    vdata = tsinfer.SampleData.from_tree_sequence(ts)
    inferred_ts = tsinfer.infer(vdata)
    inferred_ts = inferred_ts.simplify()
    dated_ts = tsdate.date(inferred_ts, mutation_rate=1e-8)
    return dated_ts

def main():
    parser = argparse.ArgumentParser(description="Infer tree sequences from simulated data")
    parser.add_argument("--sim_tree_path", required=True, help="Path to simulated tree directory")
    parser.add_argument("--output_path", required=True, help="Output directory for inferred trees")
    parser.add_argument("--output_name", required=True, help="Output directory for inferred trees")
    
    args = parser.parse_args()
    sim_tree_path = args.sim_tree_path
    output_path = args.output_path
    time_taken = infer_trees(sim_tree_path, output_path)
    with open(f"{output_path}/{args.output_name}", "w") as f:
        f.write(f"inference took {time_taken/60:.2f} minutes")

if __name__ == "__main__":
    main()
