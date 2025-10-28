#%%
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.io as pio
from itertools import combinations
import sys
sys.path.append("../../src")
from latent_embeddings import *
# %%
def dynamic_3d_plot(embeddings_df, file_name ,colors_dict, dir = "outputs/chr21/subsamples/"):
    fig = px.line_3d(embeddings_df, x="X1", y="X2", z="time (years ago)",log_z=True, color='population',line_group='sample_id',
        color_discrete_map=colors_dict
    )
    fig.update_layout(showlegend=True)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-6.5, 6.5]),
            yaxis=dict(range=[-6.5, 6.5])))
    # fig.show()
    pio.write_html(fig, file=os.path.join(dir, file_name))

    return fig

def extract_matrices(df):
    populations = np.unique(df["sample_1_population"])
    population_samples_dict = {pop: [] for pop in populations}
    sample_id_map = {}
    sample_inxs = np.unique(df["sample_1_inx"].values)
    n = len(sample_inxs)
    time_inxs = np.unique(df["time_inx"].values)
    num_timepoints = len(time_inxs)

    # Map sample_inx and time_inx to matrix indices
    sample_inx_to_idx = {inx: idx for idx, inx in enumerate(sample_inxs)}
    time_inx_to_idx = {t: idx for idx, t in enumerate(time_inxs)}

    # Assign samples to populations
    for i, sample_1_inx in enumerate(sample_inxs):
        pop = df[df["sample_1_inx"] == sample_1_inx]["sample_1_population"].values[0]
        population_samples_dict[pop].append(i)
        sample_id_map[i] = sample_1_inx
    for pop, samples in population_samples_dict.items():
        population_samples_dict[pop] = np.array(samples)

    # Initialize matrices
    N = np.zeros((n, n, num_timepoints))
    L = np.zeros((n, n, num_timepoints))

    # Use vectorized assignment
    idx1 = df["sample_1_inx"].map(sample_inx_to_idx).to_numpy()
    idx2 = df["sample_2_inx"].map(sample_inx_to_idx).to_numpy()
    tidx = df["time_inx"].map(time_inx_to_idx).to_numpy()

    N[idx1, idx2, tidx] = df["num_coal_events"].values
    L[idx1, idx2, tidx] = df["num_uncoalesced"].values

    return N, L, population_samples_dict, sample_id_map



# %%
metadata = pd.read_csv("pop_labels.csv")
counts = pd.read_parquet("outputs/chr21/all_samples/counts.parquet")
times = np.unique(counts["time"].values)
R_list = counts["num_coal_events"].values/counts["num_uncoalesced"].values
R_list = R_list[~np.isnan(R_list)]
R_list = R_list[R_list != np.inf]
R_list = R_list[R_list != 0]

alpha = np.quantile(R_list, 0.995)
populations = np.unique(metadata["population"].values)
populations = populations[populations != "W x C"]

# max_combinations = len(populations)+1
max_combinations = 3
population_combinations = []
for r in range(1, max_combinations):
    population_combinations.extend([list(comb) for comb in combinations(populations, r)])
    
colors_dict = {}
for row in metadata[["population","hex_colour"]].drop_duplicates().iterrows():
    colors_dict[row[1]["population"]] = row[1]["hex_colour"]
# %%
tot_pop_combinations = len(population_combinations)
for i,pop in enumerate(population_combinations):
    print(f"Processing population combination {i+1}/{tot_pop_combinations}: {pop}")
    k = len(pop)
    metadata_subset = metadata[metadata["population"].isin(pop)]
    sample_ids = metadata_subset["sample_id"].values
    counts_subset = counts[counts["sample_1_inx"].isin(sample_ids) & counts["sample_2_inx"].isin(sample_ids)]

    N,L,population_samples_dict, sample_id_map = extract_matrices(counts_subset)

    embeddings_through_time = compute_embeddings_through_time(N, L, times, alpha=alpha)
    embeddings_df = create_embeddings_df(embeddings_through_time, times*10, population_samples_dict)
    embeddings_df.rename(columns={"time":"time (years ago)"}, inplace=True)
    embeddings_df["sample_id"] = embeddings_df["sample_id"].map(sample_id_map)


    file_name = str(k)+ ") " + "_".join(pop) + ".html"
    dynamic_3d_plot(embeddings_df, file_name ,colors_dict)



# %%
