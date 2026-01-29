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
import subprocess
import zarr

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

# %%
vcf_path = "../results/simulations/split/trees/sim"
vcf_files = glob.glob(os.path.join(vcf_path, "*.vcf"))
path_to_singer_master = "../singer-0.1.8-beta-linux-x86_64/singer_master"
path_to_singer_tskit_conversion = "../singer-0.1.8-beta-linux-x86_64/convert_to_tskit"

def infer_tsinfer(vcf_file_path, output_path, mu=1.2e-8):
    vcf_path_prefix = vcf_file_path.removesuffix(".vcf")
    contig_id = os.path.basename(vcf_path_prefix)
    output_prefix = os.path.join(output_path, contig_id)
    vcf_to_icf_command = f"vcf2zarr explode {vcf_path_prefix}.vcf {output_prefix}.icf --force"
    icf_to_vcz_command = f"vcf2zarr encode {output_prefix}.icf {output_prefix}.vcz --force"
    subprocess.run(vcf_to_icf_command, shell=True)
    subprocess.run(icf_to_vcz_command, shell=True)

    vcz_path = os.path.join(output_prefix + ".vcz")
    vcf_zarr = zarr.load(vcz_path)
    vdata = tsinfer.VariantData(vcz_path, ancestral_state=vcf_zarr["variant_allele"][:, 0])
    inferred_ts = tsinfer.infer(vdata)
    inferred_ts = inferred_ts.simplify()
    dated_ts = tsdate.date(inferred_ts, mutation_rate=mu)
    dated_ts.dump(os.path.join(output_path, f"{contig_id}.trees"))
    return

def infer_singer(vcf_file_path, output_path, path_to_singer_master, path_to_singer_tskit_conversion, mu=1e-8, contig_length=1e6, num_posterior_samples = 50, thinning_interval=100):
    vcf_path_prefix = vcf_file_path.removesuffix(".vcf")
    contig_id = os.path.basename(vcf_path_prefix)
    output_prefix = os.path.join(output_path, contig_id)
    singer_infer_command = f"{path_to_singer_master} -vcf {vcf_path_prefix} -output {output_prefix} -m {mu} -start 0 -end {contig_length} -n {num_posterior_samples} -thin {thinning_interval}"
    convert_to_tskit_command = f"{path_to_singer_tskit_conversion} -input {output_prefix} -output {output_prefix} -start 0 -end {num_posterior_samples} -step 1"

    subprocess.run(singer_infer_command, shell=True)
    subprocess.run(convert_to_tskit_command, shell=True)
    return
# %%
output_path = "../results/simulations/split/trees/tsinfer"
vcf_file_path = vcf_files[0]
# %%
