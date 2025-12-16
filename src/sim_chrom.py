#%%
import argparse
import numpy as np
import pandas as pd
import os
import demes
import demesdraw
import msprime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import stdpopsim
import allel
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# sys.path.append('../../src')
# from plot_params import *
import itertools
import warnings
# rcParams['font.size'] = 6
# rcParams['xtick.major.width'] = 1
# rcParams['ytick.major.width'] = 1
# rcParams['xtick.major.size'] = 2
# rcParams['ytick.major.size'] = 2
# demes_dir = "../../data/main_deme_files/"
species = stdpopsim.get_species("HomSap")

def plot_deme(graph,max_time,colors_dict,log_time=True):
    fig,ax_demes = plt.subplots(figsize = (2,3))
    demesdraw.tubes(graph,ax=ax_demes,max_time=max_time,colours = colors_dict,labels="xticks-legend",log_time=log_time)
    if ax_demes.get_legend() is not None:
        ax_demes.get_legend().remove()
    return fig

def compute_PCs(vcf_path,pruning=100):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        callset = allel.read_vcf(vcf_path)
    gt = callset['calldata/GT']
    n_variants, n_samples, ploidy = gt.shape
    haplotype_matrix = gt.reshape(n_variants, n_samples * ploidy)
    haplotype_matrix = haplotype_matrix.T

    X = haplotype_matrix[:,::pruning]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    return pca_result

def sim_chrom(deme_file_path,output_dir,n,max_time):
    # sequence_lengths = [species.get_contig(f"chr{i}").length for i in range(1,23)]
    # recomb_rates = [species.get_contig(f"chr{i}").recombination_map.rate for i in range(1,23)]
    num_contigs = 10
    sequence_lengths = [1e8]*num_contigs
    recomb_rates = [1e-8]*num_contigs
    vcf_path = output_dir + "mutations.vcf"

    deme_name = os.path.basename(deme_file_path).split(".")[0]
    graph = demes.load(deme_file_path)
    demography = msprime.Demography.from_demes(graph)

    K=0
    samples = {}
    population_samples_dict = {}
    extant_populations = []
    for pop in demography.populations:
        if pop.default_sampling_time == 0:
            samples[pop.name] = n
            extant_populations.append(pop.name)
            population_samples_dict[pop.name] = np.arange(2*n*K,2*n*(K+1))
            K+=1

    print(f"Simulating {n*K} samples from {K} extant populations under {deme_name}")
    start = 0
    for i in tqdm(range(len(sequence_lengths))):
        ts = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            sequence_length=sequence_lengths[i],
            recombination_rate=recomb_rates[i],
            random_seed=42+i
        )
        ts.dump(output_dir + f"chr{i+1}.trees")
        ts = msprime.sim_mutations(ts,1e-8,random_seed=42+i)
        with open(vcf_path, "a") as f:
            ts.write_vcf(f, contig_id=i+1, position_transform = lambda x: np.round(x)+1)
        start = ts.sequence_length

    if "SS" in deme_file_path:
        cmap = plt.get_cmap("rainbow")
        colors = cmap(np.arange(K))[:,:-1]
        colors_dict = {label:mcolors.to_hex(colors[i]) for i,label in enumerate(extant_populations)}
    else:
        cmap = plt.get_cmap("Set1")
        colors = cmap(np.arange(K))[:,:-1]
        colors_dict = {label:mcolors.to_hex(colors[i]) for i,label in enumerate(extant_populations)}

    metadata = pd.DataFrame(population_samples_dict)
    metadata = pd.melt(metadata,var_name="population",value_name="sample_id")
    metadata["hex_colour"] = metadata["population"].map(colors_dict)
    metadata.to_csv(output_dir + f"metadata.csv", index=False)
    if "SS" not in deme_file_path:
        deme_fig = plot_deme(graph,max_time,colors_dict)
        deme_fig.savefig(os.path.join(output_dir,"deme.png"),dpi=500)

    # Copmute PCs and plot them
    PCs = compute_PCs(vcf_path)
    pca_fig,ax = plt.subplots(figsize=(2,2))
    for pop,sample_ids in population_samples_dict.items():
        ax.scatter(PCs[sample_ids,0],PCs[sample_ids,1],color=colors_dict[pop],s=1,label=pop)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    pca_fig.savefig(os.path.join(output_dir,"pca.png"),dpi=500)

    # return ts,deme_fig,pca_fig

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="path to demes file")
    parser.add_argument("--max_time", type=int, required=True, help="maximum timepoint")

    args = parser.parse_args()
    max_time = args.max_time
    deme_file_path = args.input
    deme_name = os.path.basename(deme_file_path).split(".")[0]
    output_dir = f"./trees/{deme_name}/"
    os.makedirs(output_dir, exist_ok=True)

    # L=2e8
    # r=1e-8
    if "SS" in deme_file_path:
        n=1
    else:
        n=25


    sim_chrom(deme_file_path, output_dir, n, max_time)
    # deme_fig.savefig(os.path.join(output_dir,"deme.png"),dpi=500)
    # pca_fig.savefig(os.path.join(output_dir,"pca.png"),dpi=500)

# %%
if __name__ == "__main__":
    main()
# %%
