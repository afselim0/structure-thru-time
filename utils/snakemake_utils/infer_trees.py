#%%
#!/usr/bin/env python3
import argparse
import tskit
import tsinfer
import tsdate
import glob
import os
import numpy as np
from pathos.multiprocessing import Pool
import subprocess
import zarr
import shutil
from time import time
from tqdm import tqdm
import msprime

script_dir = os.path.dirname(os.path.abspath(__file__))
# The pipeline's relative paths (configs, data, results) are all anchored at utils/,
# which is one level above this script since the move into snakemake_utils/.
utils_dir = os.path.dirname(script_dir)
os.chdir(utils_dir)

# ARG inference binaries live alongside the other inference tooling
singer_path = os.path.join(utils_dir, "arg_infer_utils", "singer_linux")
relate_path = os.path.join(utils_dir, "arg_infer_utils", "relate_mac")

def find_Ne(sim_ts_dir):
    ts_files = glob.glob(os.path.join(sim_ts_dir, "*.trees"))
    Ne = []
    for ts_file in ts_files:
        ts = tskit.load(ts_file)
        Ne.append(ts.diversity(mode="branch")/4)
    return np.mean(Ne)

def infer_tsinfer(vcf_file_path, output_path, output_name, mu=1.2e-8, r=1.2e-8, from_TreeSequence = True):
    vcf_path_prefix = vcf_file_path.removesuffix(".vcf")
    contig_id = os.path.basename(vcf_path_prefix)
    output_prefix = os.path.join(output_path, contig_id)
    if from_TreeSequence:
        ts = tskit.load(vcf_path_prefix + ".trees")
        ts_mutated = msprime.sim_mutations(ts,rate=mu,random_seed=int(contig_id.split("_")[-1]) + 42)
        vdata = tsinfer.SampleData.from_tree_sequence(ts_mutated)
    else:
        vcf_to_icf_command = f"vcf2zarr explode {vcf_path_prefix}.vcf {output_prefix}.icf --force"
        icf_to_vcz_command = f"vcf2zarr encode {output_prefix}.icf {output_prefix}.vcz --force"
        with open(f"{output_path}/{output_name}".removesuffix(".txt")+".log", "a") as f:
            subprocess.run(vcf_to_icf_command, shell=True, stdout = f, stderr = f)
            subprocess.run(icf_to_vcz_command, shell=True, stdout = f, stderr = f)
        vcz_path = os.path.join(output_prefix + ".vcz")
        vcf_zarr = zarr.load(vcz_path)
        vdata = tsinfer.VariantData(vcz_path, ancestral_state=vcf_zarr["variant_allele"][:, 0])
    inferred_ts = tsinfer.infer(vdata, recombination_rate=r)
    inferred_ts = inferred_ts.simplify()
    dated_ts = tsdate.variational_gamma(inferred_ts, mutation_rate=mu)
    dated_ts.dump(os.path.join(output_path, f"{contig_id}.trees"))

    if not from_TreeSequence:
        shutil.rmtree(f"{output_prefix}.icf")
        shutil.rmtree(f"{output_prefix}.vcz")
    return

def infer_singer(vcf_file_path, output_path, output_name, Ne, mu=1.2e-8, contig_length=1e6, num_posterior_samples = 50, thinning_interval=100, burnin=1000):
    vcf_path_prefix = vcf_file_path.removesuffix(".vcf")
    contig_id = os.path.basename(vcf_path_prefix)
    output_prefix = os.path.join(output_path, contig_id)
    num_burnin_samples = int(burnin/thinning_interval)
    tot_posterior_samples = num_burnin_samples + num_posterior_samples
    singer_infer_command = f"{singer_path}/singer_master -vcf {vcf_path_prefix} -output {output_prefix} -m {mu} -Ne {Ne} -start 0 -end {contig_length} -n {tot_posterior_samples} -thin {thinning_interval}"
    convert_to_tskit_command = f"{singer_path}/convert_to_tskit -input {output_prefix} -output {output_prefix} -start {num_burnin_samples} -end {tot_posterior_samples} -step 1"

    with open(f"{output_path}/{output_name}".removesuffix(".txt")+".log", "a") as f:
        subprocess.run(singer_infer_command, shell=True, stdout = f, stderr = f)
        subprocess.run(convert_to_tskit_command, shell=True, stdout = f, stderr = f)
    
    os.remove(f"{output_prefix}.log")
    os.remove(f"{output_prefix}_coordinates.txt")
    os.remove(f"{output_prefix}_start_branches_0.txt")
    os.remove(f"{output_prefix}_start_nodes_0.txt")
    os.remove(f"{output_prefix}_start_muts_0.txt")
    os.remove(f"{output_prefix}_start_recombs_0.txt")
    for i in range(num_posterior_samples):
        os.remove(f"{output_prefix}_branches_{i}.txt")
        os.remove(f"{output_prefix}_nodes_{i}.txt")
        os.remove(f"{output_prefix}_muts_{i}.txt")
        os.remove(f"{output_prefix}_recombs_{i}.txt")
    return

def infer_relate(vcf_file_path, output_path, output_name, Ne, mu=1.2e-8, r=1.2e-8, contig_length=1e6):
    vcf_path_prefix = vcf_file_path.removesuffix(".vcf")
    contig_id = os.path.basename(vcf_path_prefix)
    output_prefix = os.path.join(output_path, contig_id)

    with open(f"{output_prefix}.map", "w") as f:
        f.write(f"Position.bp.\tRate.cM.Mb.\tMap.cM.\n")
        f.write(f"0\t{100*r*contig_length}\t{0}\n")
        f.write(f"{contig_length}\t{0}\t{100*r*contig_length}")
    
    vcf_to_haps_command = f"{relate_path}/RelateFileFormats --mode ConvertFromVcf --input {vcf_path_prefix} --haps {output_prefix}.haps --sample {output_prefix}.sample"
    os.chdir(output_path)
    relative_relate_path = os.path.relpath(relate_path, start=os.getcwd())
    relate_infer_command = f"{relative_relate_path}/Relate --mode All -m {mu} --effectiveN {int(2*Ne)} --haps {contig_id}.haps --sample {contig_id}.sample --map {contig_id}.map --output {contig_id}"
    convert_to_tskit_command = f"{relative_relate_path}/RelateFileFormats --mode ConvertToTreeSequence --input {contig_id} --output {contig_id}"
    os.chdir(utils_dir)
    with open(f"{output_path}/{output_name}".removesuffix(".txt")+".log", "a") as f:
        subprocess.run(vcf_to_haps_command, shell=True, stdout = f, stderr = f)
        os.chdir(output_path)
        subprocess.run(relate_infer_command, shell=True, stdout = f, stderr = f)
        subprocess.run(convert_to_tskit_command, shell=True, stdout = f, stderr = f)
        os.chdir(utils_dir)
    
    if os.path.exists(f"{output_prefix}.trees"):
        os.remove(f"{output_prefix}.map")
        os.remove(f"{output_prefix}.haps")
        os.remove(f"{output_prefix}.sample")
        os.remove(f"{output_prefix}.anc")
        os.remove(f"{output_prefix}.mut")

    return

def infer_single(vcf_file_path, output_path, output_name, Ne, mu, r, method, contig_length = None, num_posterior_samples = None, thinning_interval = None, burnin=None):
    if method == "tsinfer":
        infer_tsinfer(vcf_file_path, output_path, output_name, mu=mu, r=r)
    elif method == "singer":
        infer_singer(vcf_file_path, output_path, output_name, Ne, mu=mu, contig_length=contig_length, num_posterior_samples=num_posterior_samples, thinning_interval=thinning_interval, burnin=burnin)
    elif method == "relate":
        infer_relate(vcf_file_path, output_path, output_name, Ne, mu=mu, r=r, contig_length=contig_length)
    else:
        raise ValueError(f"Unknown method: {method}")
    return

def infer_multiple(vcf_dir, output_path, output_name, Ne, mu, r, method, contig_length = None, num_posterior_samples = None, thinning_interval = None, burnin=None, num_threads = 1):
    # simulate_trees.py writes .trees only (its VCF output is disabled), while
    # SINGER and Relate read the .vcf text directly. Fall back to enumerating
    # the tree sequences so the tsinfer path, which loads contig_N.trees via
    # from_TreeSequence, works without VCFs on disk. The .vcf suffix is kept on
    # the paths because infer_single derives the contig id by stripping it.
    vcf_files = sorted(glob.glob(os.path.join(vcf_dir, "*.vcf")))
    if not vcf_files:
        vcf_files = [f.removesuffix(".trees") + ".vcf"
                     for f in sorted(glob.glob(os.path.join(vcf_dir, "*.trees")))]
    if not vcf_files:
        raise ValueError(f"No .vcf or .trees files found in directory: {vcf_dir}")
    os.makedirs(output_path, exist_ok=True)
    def infer_single_wrapper(vcf_file_path):
        return infer_single(vcf_file_path, output_path, output_name, Ne, mu, r, method, contig_length, num_posterior_samples, thinning_interval, burnin)
    start_time = time()
    if num_threads == 1:
        for vcf_file in tqdm(vcf_files):
            infer_single_wrapper(vcf_file)
        return time() - start_time
    else:
        with Pool(num_threads) as pool:
            list(tqdm(pool.imap(infer_single_wrapper, vcf_files), total=len(vcf_files)))
    return time() - start_time

def main():
    parser = argparse.ArgumentParser(description="Infer tree sequences from simulated data")
    parser.add_argument("--vcf_dir", required=True, help="Path to simulated tree directory")
    parser.add_argument("--output_path", required=True, help="Output directory for inferred trees")
    parser.add_argument("--method", required=True, help="Method used for inference")
    parser.add_argument("--contig_length", type=int, default=1e6, help="length of contigs")
    parser.add_argument("--num_posterior_samples", type=int, default=100, help="Number of posterior samples for singer")
    parser.add_argument("--thinning_interval", type=int, default=60, help="Thinning interval for singer")
    parser.add_argument("--burnin", type=int, default=1000, help="Burn-in period for singer")
    parser.add_argument("--mu", type=float, default=1.2e-8, help="Mutation rate")
    parser.add_argument("--r", type=float, default=1.2e-8, help="Recombination rate")
    parser.add_argument("--output_name", required=True, help="Name of the output file to save inference time")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of threads to use for inference")
    args = parser.parse_args()
    vcf_dir = args.vcf_dir
    output_path = args.output_path
    method = args.method
    contig_length = args.contig_length
    num_posterior_samples = args.num_posterior_samples
    thinning_interval = args.thinning_interval
    burnin = args.burnin
    mu = args.mu
    r = args.r
    output_name = args.output_name
    num_threads = args.num_threads
    Ne = find_Ne(vcf_dir)

    print(f"inferring trees from vcf files in {vcf_dir} using method {method}")
    time_taken = infer_multiple(vcf_dir, output_path, output_name, Ne, mu, r, method, contig_length, num_posterior_samples, thinning_interval, burnin, num_threads)

    with open(f"{output_path}/{output_name}", "w") as f:
        f.write(f"inference took {time_taken/60:.2f} minutes")

if __name__ == "__main__":
    main()
# %%
