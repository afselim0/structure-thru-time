#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../src")
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import os
from plotting_style import *
from optimization_functions import optimize_timepoint, compute_f_inverse
from tqdm import tqdm

def compute_embeddings_panmictic(Ne, link, p, n=50, N_min = 1, N_max = 1e8, init_seed=42):
    np.random.seed(init_seed)
    X_init = np.random.rand(n, 2)
    R = np.ones((n,n)) * 1/Ne
    np.fill_diagonal(R, np.nan)

    X,_,_,_ = optimize_timepoint(X_init, R, W_t=None, N_min=N_min, N_max=N_max, p=p, link=link)
    X = X - np.mean(X, axis=0) # center the embeddings
    return X
# %%
Ne_list = [1000, 5000, 10000, 20000]
# N_min_list = np.logspace(1, np.log10(12000), 100).astype(int)
N_min_list = np.linspace(1, 21000, 100).astype(int)
# n_list = np.arange(10,100,2).astype(int)
n_list = np.logspace(1, np.log10(500), 50).astype(int)
N_max = 1e8

average_pairwise_distances_Nmin = {}
average_pairwise_distances_n = {}
for Ne in Ne_list:
    average_pairwise_distances_Nmin[Ne] = []
    average_pairwise_distances_n[Ne] = []
    for N_min in tqdm(N_min_list):
        X = compute_embeddings_panmictic(Ne, link="inverse_distance", p=1, n=50, N_min=N_min, N_max=N_max)
        avg_distance = np.nanmean(pdist(X))
        average_pairwise_distances_Nmin[Ne].append(avg_distance)
    for n in tqdm(n_list):
        X = compute_embeddings_panmictic(Ne, link="inverse_distance", p=1, n=n, N_min=1, N_max=N_max)
        avg_distance = np.nanmean(pdist(X))
        average_pairwise_distances_n[Ne].append(avg_distance)
#%%
fig,ax = plt.subplots(1,2,figsize=(6,3))
for Ne in Ne_list:
    ax[0].plot(N_min_list, average_pairwise_distances_Nmin[Ne], label=f"Ne={Ne}")
    ax[1].plot(n_list, average_pairwise_distances_n[Ne], label=f"Ne={Ne}")
ax[0].set_xlabel(r"Minimum resolvable $N_e$ ($N_{\mathrm{min}}$)")
ax[0].set_ylabel("Average pairwise distance")
ax[1].set_xlabel("number of samples")
ax[1].set_ylabel("Average pairwise distance")
ax[0].legend()
# ax[1].legend()
# ax[1].set_xscale('log')
# ax[1].set_yscale('log')
ax[0].set_title("A", loc='left', fontweight='bold')
ax[1].set_title("B", loc='left', fontweight='bold')
fig.suptitle(r"$f(x) = (N_{\mathrm{min}} + x)^{-1}$")
file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../outputs/{file_name}.pdf", dpi=500)
# %%
