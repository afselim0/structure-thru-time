#%%
import matplotlib.pyplot as plt
import numpy as np
import os
os.chdir(os.path.dirname(__file__))
from plotting_style import *
import sys
from tqdm import tqdm
sys.path.append("../../../src")
from optimization_functions import optimize_timepoint

def construct_R(Ne_list, n=20):
    K = len(Ne_list)
    R_diag = np.diag(1/np.array(Ne_list))
    R = np.kron(R_diag, np.ones((n,n)))
    np.fill_diagonal(R, np.nan)
    return R

def compute_embeddings(R, link, p, n=20, N_min = 1, N_max = 1e5, X_init=None, init_seed=42):
    if X_init is None:
        np.random.seed(init_seed)
        X_init = np.random.rand(len(Ne_list)*n, 2)

    X,_,_,_ = optimize_timepoint(X_init, R, W_t=None, N_min=N_min, N_max=N_max, p=p, link=link)
    X = X - np.mean(X, axis=0) # center the embeddings
    return X
# %%
Ne_list = np.linspace(1e3, 1e4, 10).astype(int)
n=20
N_min = 1
N_max = 1e6
link_functions = [
    {"link": "inverse_distance", "p": 0.5},
    {"link": "inverse_distance", "p": 1},
    {"link": "inverse_distance", "p": 2},
    {"link": "inverse_distance", "p": 4},
    {"link": "exponential", "p": 0.5},
    {"link": "exponential", "p": 1},
    {"link": "exponential", "p": 1.5},
    {"link": "exponential", "p": 2}
]
X_dict_Nmax_1e5 = {}
X_dict_Nmax_1e6 = {}
for link_dict in tqdm(link_functions):
    link = link_dict["link"]
    p = link_dict["p"]
    R = construct_R(Ne_list, n=n)
    X_Nmax_1e5 = compute_embeddings(R, link=link, p=p, n=n, N_min=N_min, N_max=1e5)
    X_Nmax_1e6 = compute_embeddings(R, link=link, p=p, n=n, N_min=N_min, N_max=1e6)
    X_dict_Nmax_1e5[(link, p)] = X_Nmax_1e5
    X_dict_Nmax_1e6[(link, p)] = X_Nmax_1e6
# %%
colors = plt.cm.viridis(np.linspace(0, 1, len(Ne_list)))
fig = plt.figure(figsize=(6,7))
gs = fig.add_gridspec(5, 4, height_ratios=[1,1,0.3,1,1])
axs = np.empty((4,4), dtype=object)
for r in range(4):
    gs_row = r if r < 2 else r + 1
    for c in range(4):
        axs[r,c] = fig.add_subplot(gs[gs_row, c])
for i, link_dict in enumerate(link_functions):
    link = link_dict["link"]
    p = link_dict["p"]
    X_Nmax_1e5 = X_dict_Nmax_1e5[(link, p)]
    X_Nmax_1e6 = X_dict_Nmax_1e6[(link, p)]
    ax_Nmax_1e5 = axs[i//4, i%4]
    ax_Nmax_1e6 = axs[i//4 + 2, i%4]
    ax_Nmax_1e5.scatter(X_Nmax_1e5[:,0], X_Nmax_1e5[:,1], c=np.repeat(colors, n, axis=0), s=10)
    ax_Nmax_1e6.scatter(X_Nmax_1e6[:,0], X_Nmax_1e6[:,1], c=np.repeat(colors, n, axis=0), s=10)
    ax_Nmax_1e5.set_title(f"p={p}", fontsize=10)
    ax_Nmax_1e6.set_title(f"p={p}", fontsize=10)
# axs[0,0].set_ylabel(r"$f(x) = (1 + x^p)^{-1}$", fontsize=12)
# axs[2,0].set_ylabel(r"$f(x) = (1 + x^p)^{-1}$", fontsize=12)
# axs[1,0].set_ylabel(r"$f(x) = e^{-x^p}$", fontsize=12)
# axs[3,0].set_ylabel(r"$f(x) = e^{-x^p}$", fontsize=12)
fig.canvas.draw()
row0_left = axs[0,0].get_position().x0
row0_right = axs[0,3].get_position().x1
row0_top = axs[0,0].get_position().y1
row2_top = axs[2,0].get_position().y1
row_center_x = (row0_left + row0_right) / 2
fig.text(row_center_x, row0_top + 0.05, r"$N_{\text{min}} = 1,\ N_{\text{max}} = 10^5$", ha='center', va='bottom', fontsize=12)
fig.text(row_center_x, row2_top + 0.05, r"$N_{\text{min}} = 1,\ N_{\text{max}} = 10^6$", ha='center', va='bottom', fontsize=12)

fig.text(row0_left - 0.07, (axs[0,0].get_position().y0 + axs[0,0].get_position().y1) / 2, r"$f(x) = (1 + x^p)^{-1}$", ha='right', va='center', rotation=90, fontsize=12)
fig.text(row0_left - 0.07, (axs[2,0].get_position().y0 + axs[2,0].get_position().y1) / 2, r"$f(x) = (1 + x^p)^{-1}$", ha='right', va='center', rotation=90, fontsize=12)
fig.text(row0_left - 0.07, (axs[1,0].get_position().y0 + axs[1,0].get_position().y1) / 2, r"$f(x) = e^{-x^p}$", ha='right', va='center', rotation=90, fontsize=12)
fig.text(row0_left - 0.07, (axs[3,0].get_position().y0 + axs[3,0].get_position().y1) / 2, r"$f(x) = e^{-x^p}$", ha='right', va='center', rotation=90, fontsize=12)

fig.text(row0_left - 0.07, row0_top + 0.05, "A", ha='right', va='bottom', fontsize=12, fontweight='bold')
fig.text(row0_left - 0.07, row2_top + 0.05, "B", ha='right', va='bottom', fontsize=12, fontweight='bold')

# legend mapping Ne values to their viridis color, placed to the right of the grid
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[j], markersize=6, label=f"{Ne}")
    for j, Ne in enumerate(Ne_list)
]
row0_right_all = axs[0,3].get_position().x1
fig.legend(handles=legend_handles, title=r"$N_e$", loc='center left', bbox_to_anchor=(row0_right_all + 0.02, 0.5), fontsize=9, title_fontsize=10, frameon=False)

file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../outputs/{file_name}.pdf", dpi=500)

# %%
