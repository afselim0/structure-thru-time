#%%
import demes
import msprime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
import demesdraw
#%%
def setup_plotting_style():
    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5
    # -- Fonts --
    rcParams['font.size'] = 10
    rcParams['font.family'] = 'serif'
    # plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    # -- Figure size --
    rcParams['figure.figsize'] = (6, 6)
    # -- Saving Options --
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.dpi'] = 500
#%%
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
    pop_labels = np.repeat(pops, n_samples_per_pop)
    pop_labels = pd.DataFrame({
        "sample_id": np.arange(len(pop_labels)),
        "population": pop_labels
    })
    return pop_labels, pop_colors

def plot_demography(demes_file_path, pop_colors, max_time):
    setup_plotting_style()
    graph = demes.load(demes_file_path)
    pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
    fig, ax = plt.subplots(figsize=(3, 6))
    demesdraw.tubes(
        graph,
        ax=ax,
        colours=pop_colors_dict,
        max_time=max_time,
        log_time=True
    )
    return fig

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
        ploidy=1,
        random_seed=42
    )
    pca=ts.pca(num_components=num_components).factors
    sample_ids = np.arange(pca.shape[0])
    data = np.hstack((sample_ids[:,None],pca))
    pca_df = pd.DataFrame(data,columns=["sample_id"]+[f"PC{i+1}" for i in range(num_components)])
    pca_df["sample_id"] = pca_df["sample_id"].astype(int)
    return pca_df

def plot_pca(pca_df, pop_labels, pop_colors):
    setup_plotting_style()
    merged_df = pca_df.merge(pop_labels, on="sample_id").merge(pop_colors, on="population")
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
# %%
demes_file_path = "../../../data/demes/main/split.yaml"
n_samples_per_pop = 20
pop_labels, pop_colors = preprocess_demography(demes_file_path, n_samples_per_pop)
fig = plot_demography(demes_file_path, pop_colors, max_time=20000)
pca_df = run_pca(demes_file_path, n_samples_per_pop, sequence_length=1e7, recombination_rate=1e-8, num_components=2)
pca_fig = plot_pca(pca_df, pop_labels, pop_colors)
# %%
