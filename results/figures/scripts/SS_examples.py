#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from plotting_style import *
import sys
import os
import demes
import msprime
filename = os.path.basename(__file__).split(".")[0]
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils")
from PlotEmbeddings import *
from OptimizationFunctions import procrustes_align_trajectories_df
import yaml
def find_embeddings_file(SS, p, m):
    m_mantissa = str(m).split(".")[1]
    data_dir = f"../../../results/simulations/outputs/SS/SS_{SS}d_N100_m{m_mantissa}/data"
    files = os.listdir(data_dir)
    for file in files:
        if f"p{p}" in file and file.startswith("embeddings"):
            return os.path.join(data_dir, file)
    return None
#%%
# SS_1d_p1_m05_embeddings = pd.read_parquet(find_embeddings_file(SS=1, p=1, m=0.05))
# SS_1d_p1_m01_embeddings = pd.read_parquet(find_embeddings_file(SS=1, p=1, m=0.01))
# SS_1d_p1_m001_embeddings = pd.read_parquet(find_embeddings_file(SS=1, p=1, m=0.001))

# SS_1d_p2_m05_embeddings = pd.read_parquet(find_embeddings_file(SS=1, p=2, m=0.05))
# SS_1d_p2_m01_embeddings = pd.read_parquet(find_embeddings_file(SS=1, p=2, m=0.01))
# SS_1d_p2_m001_embeddings = pd.read_parquet(find_embeddings_file(SS=1, p=2, m=0.001))

SS_2d_p1_m05_embeddings = pd.read_parquet(find_embeddings_file(SS=2, p=1, m=0.05))
SS_2d_p1_m01_embeddings = pd.read_parquet(find_embeddings_file(SS=2, p=1, m=0.01))
SS_2d_p1_m001_embeddings = pd.read_parquet(find_embeddings_file(SS=2, p=1, m=0.001))

SS_2d_p2_m05_embeddings = pd.read_parquet(find_embeddings_file(SS=2, p=2, m=0.05))
SS_2d_p2_m01_embeddings = pd.read_parquet(find_embeddings_file(SS=2, p=2, m=0.01))
SS_2d_p2_m001_embeddings = pd.read_parquet(find_embeddings_file(SS=2, p=2, m=0.001))

SS_1d_pop_labels = "../../../results/simulations/outputs/SS/SS_1d_N100_m05/data/pop_labels.csv"
SS_1d_pop_colors = "../../../results/simulations/outputs/SS/SS_1d_N100_m05/data/pop_colors.csv"

SS_2d_pop_labels = "../../../results/simulations/outputs/SS/SS_2d_N100_m05/data/pop_labels.csv"
SS_2d_pop_colors = "../../../results/simulations/outputs/SS/SS_2d_N100_m05/data/pop_colors.csv"
# %%
