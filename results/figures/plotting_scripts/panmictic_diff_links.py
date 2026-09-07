#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../src")
import os
from results.figures.plotting_style import *
from optimization_functions import optimize_timepoint
def compute_embeddings_panmictic(Ne, link, p, n=50, N_min = 1, N_max = 1e8, init_seed=42):
    np.random.seed(init_seed)
    X_init = np.random.rand(n, 2)
    R = np.ones((n,n)) * 1/Ne
    np.fill_diagonal(R, np.nan)

    X,_,_,_ = optimize_timepoint(X_init, R, W_t=None, N_min=N_min, N_max=N_max, p=p, link=link)
    X = X - np.mean(X, axis=0) # center the embeddings
    return X
# %%
Ne = 1e4
link_functions = [
    {"link": "inverse_distance", "p": 0.1},
    {"link": "inverse_distance", "p": 0.5},
    {"link": "inverse_distance", "p": 1},
    {"link": "inverse_distance", "p": 2},
    {"link": "exponential", "p": 0.5},
    {"link": "exponential", "p": 1},
    {"link": "exponential", "p": 2},
    {"link": "exponential", "p": 4}
]
fig, axs = plt.subplots(2, 4, figsize = (6,3.5))
for i, link_dict in enumerate(link_functions):
    link = link_dict["link"]
    p = link_dict["p"]
    X = compute_embeddings_panmictic(Ne, link, p)
    ax = axs[i//4, i%4]
    ax.scatter(X[:,0], X[:,1], s=10)
    ax.set_title(f"p={p}", fontsize=10)
fig.text(-0.01, 0.78, r"$f(x) = (1 + x^p)^{-1}$", ha='center', va='center', fontsize=12, rotation=90)
fig.text(-0.01, 0.28, r"$f(x) = e^{-x^p}$", ha='center', va='center', fontsize=12, rotation=90)

file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../outputs/{file_name}.pdf", dpi=500)
# %%