#%%
import sys
sys.path.append("../../../src")
import numpy as np
import pandas as pd
import demes
import demesdraw
import os
import matplotlib.pyplot as plt
from plot_params import *
from latent_embeddings import *
import seaborn as sns
from matplotlib.gridspec import GridSpec

def plot_deme(deme_file_path,colors_dict,max_time,ax):
    graph = demes.load(deme_file_path)
    demesdraw.tubes(graph,ax=ax,max_time=max_time,colours = colors_dict,labels="xticks-legend",log_time=True)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    return fig

def df_to_matrix(counts_df):
    n = np.max(counts_df["sample_1_inx"]) + 1
    num_timepoints = np.max(counts_df["time_inx"]) + 1
    N = np.zeros((n, n, num_timepoints), dtype=int)
    L = np.zeros((n, n, num_timepoints), dtype=int)

    # Use pandas pivot for fast reshaping
    N_pivot = counts_df.pivot_table(
        index="sample_1_inx", columns="sample_2_inx", values="num_coal_events", aggfunc="first"
    )
    L_pivot = counts_df.pivot_table(
        index="sample_1_inx", columns="sample_2_inx", values="num_uncoalesced", aggfunc="first"
    )

    for k in range(num_timepoints):
        df_k = counts_df[counts_df["time_inx"] == k]
        N_k = df_k.pivot(index="sample_1_inx", columns="sample_2_inx", values="num_coal_events").fillna(0).values
        L_k = df_k.pivot(index="sample_1_inx", columns="sample_2_inx", values="num_uncoalesced").fillna(0).values
        # Only fill upper triangle (i < j)
        for i in range(n):
            for j in range(i+1, n):
                N[i, j, k] = N_k[i, j]
                L[i, j, k] = L_k[i, j]
    return N, L



# %%
deme_name = "zigzag"

# Load data
deme_file_path = f"../../../data/demes_tskit/{deme_name}.yaml"
metadata = pd.read_parquet(f"../../../data/sim_results/{deme_name}/metadata.parquet")
counts_df_sim = pd.read_parquet(f"../../../data/sim_results/{deme_name}/counts.parquet")
counts_df_infer = pd.read_parquet(f"../../../data/results/{deme_name}_infer/counts.parquet")
pca_df = pd.read_parquet(f"../../../data/sim_results/{deme_name}/pca.parquet")
colors_dict = metadata.set_index("population")["hex_colour"].to_dict()
times = counts_df_sim["time"].unique()
n = metadata["sample_id"].nunique()

# Demes Fig
max_time = counts_df_sim["time"].max()
demes_fig = plot_deme(deme_file_path,colors_dict,max_time)
#%%
# PCA Fig
sns.scatterplot(data=pca_df,x="X1",y="X2",hue="population",palette=colors_dict,legend=False)
plt.xlabel("PC1")
plt.ylabel("PC2")
# %%
np.random.seed(123)
X_init = np.random.rand(n, 2)
N_sim,L_sim = df_to_matrix(counts_df_sim)
N_infer,L_infer = df_to_matrix(counts_df_infer)
X_over_time_sim = compute_embeddings_through_time(N_sim, L_sim, times, X_init=X_init)
X_over_time_infer = compute_embeddings_through_time(N_infer, L_infer, times, X_init=X_init)
# # %%
inx = 80
fig,ax = plt.subplots(1,2,figsize=(8,4))
ax[0].scatter(X_over_time_sim[:40,0,inx], X_over_time_sim[:40,1,inx], c=colors_dict["generic"], s=10)
# ax[0].scatter(X_over_time_sim[40:,0,inx], X_over_time_sim[40:,1,inx], c=colors_dict["B"], s=10)

ax[1].scatter(X_over_time_infer[:40,0,inx], X_over_time_infer[:40,1,inx], c=colors_dict["generic"], s=10)
# ax[1].scatter(X_over_time_infer[40:,0,inx], X_over_time_infer[40:,1,inx], c=colors_dict["B"], s=10)

# %%
# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Time axis (z-axis)
time = np.arange(100)

# Plot first 40 samples (Group A)
for i in range(40):
    ax.plot(
        X_over_time_infer[i, 0, :],  # x-coordinates over time
        X_over_time_infer[i, 1, :],  # y-coordinates over time
        time,                     # z-axis (time)
        color=colors_dict["generic"], alpha=0.5, linewidth=1
    )

# Plot next 40 samples (Group B)
# for i in range(40, 80):
#     ax.plot(
#         X_over_time_infer[i, 0, :],
#         X_over_time_infer[i, 1, :],
#         time,
#         color=colors_dict["B"], alpha=0.5, linewidth=1
#     )

ax.view_init(elev=10, azim=-75)
# Labels and title
ax.set_xlabel('X1', fontsize=12)
ax.set_ylabel('X2', fontsize=12)
ax.set_zlabel('time', fontsize=12)
# %%
demes_list = ["zigzag","split","split_mig_Nm_1","admix_05_05"]
# deme_names = ["Split","Zigzag","Split with migration (Nm=1)","Admixture"]
fig = plt.figure(figsize=(10.5, 12))

# Create a 4x4 grid with custom widths
gs = GridSpec(
    nrows=4, ncols=4,
    width_ratios=[1.5, 3, 3, 3],  # First column narrower
    height_ratios=[3, 3, 3, 3],    # Uniform row heights
    wspace=0.3, hspace=0.3         # Adjust spacing between subplots
)

# Generate subplots
for row,deme in enumerate(demes_list):
    deme_file_path = f"../../../data/demes_tskit/{deme}.yaml"
    metadata = pd.read_parquet(f"../../../data/sim_results/{deme}/metadata.parquet")
    counts_df_sim = pd.read_parquet(f"../../../data/sim_results/{deme}/counts.parquet")
    counts_df_infer = pd.read_parquet(f"../../../data/results/{deme}_infer/counts.parquet")
    colors_dict = metadata.set_index("population")["hex_colour"].to_dict()
    K = metadata["population"].nunique()
    pops = metadata["population"].unique()
    n=40
    for col in range(4):
        if col == 0:
            ax = fig.add_subplot(gs[row, col])
            if row == 0 :
                ax.set_title("Simulated Demography", fontsize=14)
            # Plot demes
            max_time = counts_df_sim["time"].max()
            plot_deme(deme_file_path, colors_dict, max_time, ax=ax)
            # ax.set_title(deme_names[row], fontsize=12)
        if col == 1:
            ax = fig.add_subplot(gs[row, col])
            if row == 0:
                ax.set_title("PCA", fontsize=14)
            # Plot PCA
            pca_df = pd.read_parquet(f"../../../data/sim_results/{deme}/pca.parquet")
            sns.scatterplot(data=pca_df, x="X1", y="X2", hue="population", palette=colors_dict, ax=ax, legend=False)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2",labelpad=-5)
        if col == 2:
            ax = fig.add_subplot(gs[row, col], projection='3d')
            if row == 0:
                ax.set_title("Embeddings True Trees", fontsize=14)
            # Plot sim embeddings
            np.random.seed(123)
            X_init = np.random.rand(metadata["sample_id"].nunique(), 2)
            N_sim, L_sim = df_to_matrix(counts_df_sim)
            N_infer, L_infer = df_to_matrix(counts_df_infer)
            times = counts_df_sim["time"].unique()
            X_over_time_sim = compute_embeddings_through_time(N_sim, L_sim, times, X_init=X_init)
            X_over_time_infer = compute_embeddings_through_time(N_infer, L_infer, times, X_init=X_init)
            # Use log10 for z axis
            log_times = np.log10(times + 1e-8)  # add small value to avoid log(0)
            for k, pop in enumerate(pops):
                inx = metadata[metadata["population"] == pop].index
                for i in inx:
                    ax.plot(
                        X_over_time_sim[i, 0, :],  # x-coordinates over time
                        X_over_time_sim[i, 1, :],  # y-coordinates over time
                        log_times,                  # z-axis (log scale)
                        color=colors_dict[pop], alpha=0.5, linewidth=1
                    )
            ax.view_init(elev=10, azim=-75)
            ax.set_xlabel('X1', fontsize=12)
            ax.set_ylabel('X2', fontsize=12)
            ax.set_zlabel('log10(time)', fontsize=12)


        if col==3:
            ax = fig.add_subplot(gs[row, col], projection='3d')
            if row == 0:
                ax.set_title("Embeddings Inferred Trees", fontsize=14)
            log_times = np.log10(times + 1e-8)  # add small value to avoid log(0)
            for k, pop in enumerate(pops):
                inx = metadata[metadata["population"] == pop].index
                for i in inx:
                    ax.plot(
                        X_over_time_infer[i, 0, :],  # x-coordinates over time
                        X_over_time_infer[i, 1, :],  # y-coordinates over time
                        log_times,                   # z-axis (log scale)
                        color=colors_dict[pop], alpha=0.5, linewidth=1
                    )
            ax.view_init(elev=10, azim=-75)
            ax.set_xlabel('X1', fontsize=12)
            ax.set_ylabel('X2', fontsize=12)
            ax.set_zlabel('log10(time)', fontsize=12)



            

# %%
fig.savefig("Fig2.png", dpi=500, bbox_inches='tight')
# %%
