#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))
from plotting_style import *
import sys
import msprime
import demes
from tqdm import tqdm
sys.path.append("../../../src")
sys.path.append("../../../utils")
from compute_rates_true import compute_true_rates, expand_true_rates
from preprocess_demography import preprocess_demography
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
from PlotEmbeddings import PlotEmbeddings
from optimization_functions import procrustes_align_trajectories_df

def pop_split(N0,N1,N2,t_split,t_mig_start=0,t_mig_end=0,m=0):
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="Ancestral")

    if m!=0:
        demography.add_symmetric_migration_rate_change(time=t_mig_start,populations=['A','B'],rate=m)
        demography.add_symmetric_migration_rate_change(time=t_mig_end,populations=['A','B'],rate=0)

    demography.sort_events()
    
    return demography
#%%
output_path = "../misc_outputs/"
os.makedirs(output_path, exist_ok=True)
max_time = 3000
delta_t = 1
N_max = 1e6
N_min = 1
Ne = 1e4
n_samples_per_pop = 20
Nm_list = np.array([1e-3, 1e-2, 1e-1, 1])
m_list = Nm_list / Ne
split_time = 2000
early_mig_end_time = 500
deep_mig_end_time = 1500
#%%
for m in m_list:
    early_migration_file = f"{output_path}early_migration_{m}.yaml"
    deep_migration_file = f"{output_path}deep_migration_{m}.yaml"
    early_migration_deme = pop_split(Ne,Ne,Ne,split_time,t_mig_start=0,t_mig_end=early_mig_end_time,m=m).to_demes()
    deep_migration_deme = pop_split(Ne,Ne,Ne,split_time,t_mig_start=deep_mig_end_time,t_mig_end=split_time,m=m).to_demes()
    demes.dump(early_migration_deme, early_migration_file)
    demes.dump(deep_migration_deme, deep_migration_file)
    pop_labels, pop_colors = preprocess_demography(early_migration_file, n_samples_per_pop=n_samples_per_pop)

    true_rates_early = compute_true_rates(early_migration_file, max_time, delta_t)
    true_rates_early = expand_true_rates(true_rates_early, pop_labels, phased = True)

    true_rates_deep = compute_true_rates(deep_migration_file, max_time, delta_t)
    true_rates_deep = expand_true_rates(true_rates_deep, pop_labels, phased = True)

    LSE_early = LatentSpaceEmbeddings(true_rates_early, N_min=N_min, N_max=N_max)
    LSE_deep = LatentSpaceEmbeddings(true_rates_deep, N_min=N_min, N_max=N_max)

    X_early, loss = LSE_early.compute_embeddings_through_time()
    X_deep, loss = LSE_deep.compute_embeddings_through_time()
    embeddings_df_early = LSE_early.get_embeddings_df()
    embeddings_df_deep = LSE_deep.get_embeddings_df()
    embeddings_df_early.to_parquet(f"{output_path}embeddings_early_mig_{m}.parquet")
    embeddings_df_deep.to_parquet(f"{output_path}embeddings_deep_mig_{m}.parquet")

    plotter_early = PlotEmbeddings(embeddings_df = embeddings_df_early, labels_df = pop_labels, colors_df = pop_colors)
    plotter_deep = PlotEmbeddings(embeddings_df = embeddings_df_deep, labels_df = pop_labels, colors_df = pop_colors)
    plotter_early.init_plot(time_scale = "linear")
    plotter_deep.init_plot(time_scale = "linear")
    plotter_early.add_sample_trajectories(alpha = 0.8)
    plotter_deep.add_sample_trajectories(alpha = 0.8)
    fig_3d_early = plotter_early.fig
    fig_3d_deep = plotter_deep.fig
    fig_3d_early.update_layout(showlegend=False)
    fig_3d_deep.update_layout(showlegend=False)
    plotter_early.save_static(f"../misc_figs/early_mig_{m}", format = "png", width = 800, height = 800)
    plotter_deep.save_static(f"../misc_figs/deep_mig_{m}", format = "png", width = 800, height = 800)
#%%
target_embeddings = pd.read_parquet(f"{output_path}embeddings_early_mig_{m_list[0]}.parquet")
fig,axs = plt.subplots(2,4,figsize=(6, 3))
for i, m in enumerate(m_list):
    # embeddings_df_early = pd.read_parquet(f"{output_path}embeddings_early_mig_{m}.parquet")
    # embeddings_df_deep = pd.read_parquet(f"{output_path}embeddings_deep_mig_{m}.parquet")
    # embeddings_df_early = procrustes_align_trajectories_df(target_embeddings, embeddings_df_early, allow_reflections=True)
    # embeddings_df_deep = procrustes_align_trajectories_df(target_embeddings, embeddings_df_deep, allow_reflections=True)

    # X_early = embeddings_df_early[["X1","X2"]].values
    # X_deep = embeddings_df_deep[["X1","X2"]].values

    # plotter_early = PlotEmbeddings(embeddings_df = embeddings_df_early, labels_df = pop_labels, colors_df = pop_colors)
    # plotter_deep = PlotEmbeddings(embeddings_df = embeddings_df_deep, labels_df = pop_labels, colors_df = pop_colors)
    # plotter_early.init_plot(time_scale = "linear")
    # plotter_deep.init_plot(time_scale = "linear")
    # plotter_early.add_sample_trajectories(alpha = 0.8)
    # plotter_deep.add_sample_trajectories(alpha = 0.8)
    # fig_3d_early = plotter_early.fig
    # fig_3d_deep = plotter_deep.fig
    # fig_3d_early.update_layout(showlegend=False)
    # fig_3d_deep.update_layout(showlegend=False)
    # plotter_early.save_static(f"../misc_figs/early_mig_{m}", format = "png", width = 800, height = 800)
    # plotter_deep.save_static(f"../misc_figs/deep_mig_{m}", format = "png", width = 800, height = 800)
    
    img_early = plt.imread(f"../misc_figs/early_mig_{m}.png")
    img_deep = plt.imread(f"../misc_figs/deep_mig_{m}.png")
    axs[0, i].spines['left'].set_visible(False)
    axs[0, i].spines['bottom'].set_visible(False)
    axs[0, i].tick_params(left=False, bottom=False)
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])
    axs[0, i].imshow(img_early, aspect = "equal")
    axs[0, i].set_title(f"$Nm = {m*Ne}$", fontsize = 8)
    # axs[1, i].set_title(f"$Nm = {m*Ne}$", fontsize = 8)
    axs[1, i].spines['left'].set_visible(False)
    axs[1, i].spines['bottom'].set_visible(False)
    axs[1, i].tick_params(left=False, bottom=False)
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])
    axs[1, i].imshow(img_deep, aspect = "equal")


fig.text(axs[0,0].yaxis.label.get_position()[0]-0.01, 1.01, "A) Recent migration", fontweight="bold", fontsize = 11)
fig.text(axs[0,1].yaxis.label.get_position()[0]-0.01, 0.5, "B) Soft split", fontweight="bold", fontsize = 11)

filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../outputs/{filename}.pdf")
# %%
