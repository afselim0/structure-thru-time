#%%
import os
import numpy as np
import pandas as pd
import sys
import demes
import msprime
sys.path.append("../../src")
from latent_embeddings import *
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

def dynamic_3d_plot(embeddings_df, output ,colors_dict):
    fig = px.line_3d(embeddings_df, x="X1", y="X2", z="time",log_z=True, color='population',line_group='sample_id',
        color_discrete_map=colors_dict
    )
    x1 = embeddings_df["X1"].values
    x2 = embeddings_df["X2"].values

    min_value = min(np.min(x1),np.min(x2)) * 1.05
    max_value = max(np.max(x1),np.max(x2)) * 1.05

    fig.update_layout(showlegend=True)

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[min_value, max_value]),
            yaxis=dict(range=[min_value, max_value]),
        )
    )

    pio.write_html(fig, file=os.path.join(output,"dynamic_3d_plot.html"))

    return fig

def static_3d_plot(embeddings_df, output, colors_dict, x_view=2,y_view=2, z_view=1):
    fig = go.Figure()
    sample_ids = np.unique(embeddings_df.sample_id.values)

    for sample_id in sample_ids:
        df_subset = embeddings_df[embeddings_df["sample_id"] == sample_id]
        fig.add_trace(go.Scatter3d(
            x=df_subset["X1"],
            y=df_subset["X2"],
            z=df_subset["time"],
            mode='lines',
            line=dict(width=0.5, color=colors_dict[df_subset["population"].values[0]]),
            name=f"Sample {sample_id}",
            opacity= 1
        ))

    # Get combined min/max from X1 and X2 columns
    xy_min = min(embeddings_df["X1"].min(), embeddings_df["X2"].min())
    xy_max = max(embeddings_df["X1"].max(), embeddings_df["X2"].max())

    # Add 10% padding for margins
    padding = 0.1 * (xy_max - xy_min)
    xy_range = [xy_min - padding, xy_max + padding]

    fig.update_layout(showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            xaxis_title='X1',
            yaxis_title='X2',
            zaxis_title='Time (generations ago)',
            zaxis_type='log'
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xy_range),  # Same range for X
            yaxis=dict(range=xy_range),  # Same range for Y
            aspectmode='manual',  # Prevents auto-scaling
            aspectratio=dict(x=1, y=1, z=0.7),  # X/Y equal, Z scaled down
        )
    )

    fig.update_layout(
        width=600,
        height=600,
        scene_camera=dict(
            eye=dict(x=x_view, y=y_view, z=z_view),  # Diagonal view
        )
    )

    fig.write_image(os.path.join(output,"static_3d_plot.png"), scale=3)
    
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

def recompute_embeddings(deme, recompute = True, X_init=None, alpha=None, type="sim"):
    if not recompute:
        embeddings_df = pd.read_parquet(f"outputs/{type}/{deme}/embeddings.parquet")
        output_path = f"outputs/{type}/{deme}/"
        colors_dict = {}
        metadata = pd.read_parquet(f"outputs/sim/{deme}/metadata.parquet")
        for row in metadata[["population","hex_colour"]].drop_duplicates().iterrows():
            colors_dict[row[1]["population"]] = row[1]["hex_colour"]
        X_init = embeddings_df[embeddings_df["time"]==min(embeddings_df["time"])][["X1","X2"]].values
        return embeddings_df, output_path, colors_dict, X_init

    n= 40 if "SS" not in deme else 2
    counts_df = pd.read_parquet(f"outputs/{type}/{deme}/counts.parquet")
    metadata = pd.read_parquet(f"outputs/{type}/{deme}/metadata.parquet")
    demes_file_path = f"deme_files/{deme}.yaml"
    output_path = f"outputs/{type}/{deme}/"
    colors_dict = {}
    for row in metadata[["population","hex_colour"]].drop_duplicates().iterrows():
        colors_dict[row[1]["population"]] = row[1]["hex_colour"]
    graph = demes.load(demes_file_path)
    demography = msprime.Demography.from_demes(graph)
    K=0
    samples = {}
    population_samples_dict = {}
    extant_populations = []
    for pop in demography.populations:
        if pop.default_sampling_time == 0:
            samples[pop.name] = n
            extant_populations.append(pop.name)
            population_samples_dict[pop.name] = np.arange(n*K,n*(K+1))
            K+=1

    N,L,_,_ = extract_matrices(counts_df)
    times = np.unique(counts_df["time"].values)

    embeddings_through_time = compute_embeddings_through_time(N, L, times, X_init=X_init, alpha=alpha)
    embeddings_df = create_embeddings_df(embeddings_through_time, times, population_samples_dict)

    embeddings_df.to_parquet(os.path.join(output_path,"embeddings.parquet"))
    X_init = embeddings_df[embeddings_df["time"]==min(times)][["X1","X2"]].values
    return embeddings_df, output_path, colors_dict, X_init
#%%
# Matching SS model results to each other for comparison 
## Using the same alpha and X_init for all models
deme = "SS_N100_m_005"
embeddings_df, output_path, colors_dict, X_init = recompute_embeddings(deme, recompute=True, alpha = 0.05)

dynamic_fig = dynamic_3d_plot(embeddings_df,output_path,colors_dict)
static_fig = static_3d_plot(embeddings_df,output_path,colors_dict,x_view=-2,y_view=2)
static_fig

deme = "SS_N100_m_001"
embeddings_df, output_path, colors_dict, X_init = recompute_embeddings(deme, recompute=True, alpha = 0.05, X_init=X_init)

dynamic_fig = dynamic_3d_plot(embeddings_df,output_path,colors_dict)
static_fig = static_3d_plot(embeddings_df,output_path,colors_dict,x_view=-2,y_view=2)
static_fig
deme = "SS_N100_m_0005"
embeddings_df, output_path, colors_dict, X_init = recompute_embeddings(deme, recompute=True, alpha = 0.05, X_init=X_init)

dynamic_fig = dynamic_3d_plot(embeddings_df,output_path,colors_dict)
static_fig = static_3d_plot(embeddings_df,output_path,colors_dict,x_view=-2,y_view=2)
static_fig
# %%
# admixture
deme = "admix_01_09"
embeddings_df, output_path, colors_dict, _ = recompute_embeddings(deme, recompute=False)

dynamic_fig = dynamic_3d_plot(embeddings_df,output_path,colors_dict)
static_fig = static_3d_plot(embeddings_df,output_path,colors_dict,x_view=-0.4,y_view=2)
static_fig

deme = "admix_03_07"
embeddings_df, output_path, colors_dict, _ = recompute_embeddings(deme, recompute=False)

dynamic_fig = dynamic_3d_plot(embeddings_df,output_path,colors_dict)
static_fig = static_3d_plot(embeddings_df,output_path,colors_dict,x_view=0.75,y_view=2)
static_fig

deme = "admix_05_05"
embeddings_df, output_path, colors_dict, _ = recompute_embeddings(deme, recompute=False)

dynamic_fig = dynamic_3d_plot(embeddings_df,output_path,colors_dict)
static_fig = static_3d_plot(embeddings_df,output_path,colors_dict,x_view=0.75,y_view=2)
static_fig

deme = "admix_05_05"
embeddings_df, output_path, colors_dict, _ = recompute_embeddings(deme, recompute=False, type="infer")

dynamic_fig = dynamic_3d_plot(embeddings_df,output_path,colors_dict)
static_fig = static_3d_plot(embeddings_df,output_path,colors_dict,x_view=0.75,y_view=2)
static_fig
# %%
