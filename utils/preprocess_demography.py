import os
import argparse
import demes
import msprime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def preprocess_demography(demes_file_path,n_samples_per_pop):
    graph = demes.load(demes_file_path)
    demography = msprime.Demography.from_demes(graph)
    pops = [pop.name for pop in demography.populations if pop.default_sampling_time == 0]
    if len(pops) <= 9:
        cmap = plt.get_cmap("Set1")
        pop_colors_dict = {pop:mcolors.to_hex(cmap(i)[:3]) for i,pop in enumerate(pops)}
    else:
        cmap = plt.get_cmap("rainbow")
        color_idx = np.linspace(0,1,len(pops))
        pop_colors_dict = {pop:mcolors.to_hex(cmap(color_idx[i])[:3]) for i,pop in enumerate(pops)}
    pop_colors = pd.DataFrame({
        "population": list(pop_colors_dict.keys()),
        "color": list(pop_colors_dict.values())
    })
    pop_labels = np.repeat(pops, 2*n_samples_per_pop) # diploid samples
    pop_labels = pd.DataFrame({
        "sample_id": np.arange(len(pop_labels)),
        "population": pop_labels
    })
    return pop_labels, pop_colors

def main():
    parser = argparse.ArgumentParser(description="Preprocess demography from a demes file")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for preprocessed demography data", type=str)
    parser.add_argument("--n", required=True, help="Number of diploid samples per population", type=int)

    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    output_path = args.output_path
    n = args.n
    pop_labels, pop_colors = preprocess_demography(demes_file_path=demes_file_path,n_samples_per_pop=n)
    pop_colors.to_csv(os.path.join(output_path,"pop_colors.csv"))
    pop_labels.to_csv(os.path.join(output_path,"pop_labels.csv"))

if __name__ == "__main__":
    main()