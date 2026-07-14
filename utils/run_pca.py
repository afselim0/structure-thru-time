import argparse
from plotting_style import setup_plotting_style
import demes
import msprime
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def run_pca(demes_file_path,n_samples_per_pop,sequence_length=1e7,recombination_rate=1e-8,num_components=2):
    graph = demes.load(demes_file_path)
    demography = msprime.Demography.from_demes(graph)
    pops = [pop.name for pop in demography.populations if pop.default_sampling_time == 0]
    samples = {pop: n_samples_per_pop for pop in pops}
    ts = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        ploidy=2,
        random_seed=42
    )
    pca=ts.pca(num_components=num_components).factors
    sample_ids = np.arange(pca.shape[0])
    data = np.hstack((sample_ids[:,None],pca))
    pca_df = pd.DataFrame(data,columns=["sample_id"]+[f"PC{i+1}" for i in range(num_components)])
    pca_df["sample_id"] = pca_df["sample_id"].astype(int)
    return pca_df

def plot_pca(pca_df, pop_labels, pop_colors, id_label="sample_id"):
    setup_plotting_style()
    pc_cols = [c for c in pca_df.columns if c.startswith("PC")]
    # When unphased, average the per-haplotype PCA coordinates within each
    # individual so that one point is plotted per diploid individual.
    if id_label != "sample_id":
        pca_df = (
            pca_df.merge(pop_labels[["sample_id", id_label]], on="sample_id")
            .groupby(id_label, as_index=False)[pc_cols].mean()
        )
    label_cols = pop_labels[[id_label, "population"]].drop_duplicates(subset=id_label)
    merged_df = pca_df.merge(label_cols, on=id_label).merge(pop_colors, on="population")
    fig, ax = plt.subplots(figsize=(6,6))
    pops = merged_df['population'].unique()
    for pop in pops:
        pop_data = merged_df[merged_df['population'] == pop]
        ax.scatter(
            pop_data['PC1'],
            pop_data['PC2'],
            label=pop,
            color=pop_data['color'].iloc[0],
            s=10
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(pops) <= 10:
        ax.legend(title="Population", bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig

def main():
    parser = argparse.ArgumentParser(description="Run PCA on simulated genetic data")
    parser.add_argument("--demes_file_path", required=True, help="Path to the demes YAML file", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for PCA results", type=str)
    parser.add_argument("--colors", required=True, help="path to colors file", type=str)
    parser.add_argument("--labels", required=True, help="path to labels file", type=str)
    parser.add_argument("--n", required=True, help="Number of samples per population", type=int)
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true", help="If set, PCA coordinates are plotted per haplotype (sample_id). Otherwise they are averaged per diploid individual (individual_id)")

    args = parser.parse_args()
    demes_file_path = args.demes_file_path
    output_path = args.output_path
    n = args.n
    colors_path = args.colors
    labels_path = args.labels
    colors = pd.read_csv(colors_path)
    labels = pd.read_csv(labels_path)
    per_haplotype = args.per_haplotype
    id_label = "sample_id" if per_haplotype else "individual_id"

    sequence_length = 1e7
    recombination_rate = 1e-8
    num_components = 2

    pca_df = run_pca(demes_file_path, n, sequence_length, recombination_rate, num_components)
    pca_fig = plot_pca(pca_df,labels,colors,id_label=id_label)

    pca_fig.savefig(os.path.join(output_path, "pca.png"), dpi=500)
    # pca_df.to_parquet(os.path.join(output_path, "pca_results.parquet"))

if __name__ == "__main__":
    main()