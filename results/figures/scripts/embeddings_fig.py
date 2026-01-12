#%%
from utils import setup_plotting_style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
sys.path.append("../../../src")
from PlotEmbeddings import *
import demes
import demesdraw
import plotly.graph_objects as go
setup_plotting_style()

def plot_demography(demes_file_path, pop_colors, max_time, ax):
    graph = demes.load(demes_file_path)
    pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
    demesdraw.tubes(
        graph,
        ax=ax,
        colours=pop_colors_dict,
        max_time=max_time,
        log_time=True
    )


# #%%
# parser = argparse.ArgumentParser(description="Plot embeddings and demography")
# parser.add_argument("--demes_file", required=True, help="List of demography examples to plit")

# # deme_names = parser.parse_args().deme_names.split(",")
# deme_name = parser.parse_args().deme_name
#%%
demes_file_path = "../../../data/demes/main/split.yaml"
deme_name = os.path.basename(demes_file_path).rstrip(".yaml")
embeddings_df = pd.read_parquet(f"../../simulations/outputs/{deme_name}/data/embeddings_sim.parquet")
pop_colors = pd.read_parquet(f"../../simulations/outputs/{deme_name}/data/pop_colors.parquet")
max_time = embeddings_df["time"].max()
fig, axs = plt.subplots(1,2,figsize=(6,3),constrained_layout=True,gridspec_kw={"width_ratios": [1, 3]}, dpi=500)
plot_demography(demes_file_path, pop_colors, max_time, ax=axs[0])


# %%
def static_3d_plot(embeddings_df, colors_dict, x_view=2,y_view=2, z_view=1):
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