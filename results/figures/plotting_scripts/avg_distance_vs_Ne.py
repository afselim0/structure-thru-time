#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../../src")
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import os
from results.figures.plotting_style import *
from OptimizationFunctions import optimize_timepoint, LinkFunction
from tqdm import tqdm

# Relative paths below are anchored at this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def hull_circle_radius(points):
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points)
    hp = points[hull.vertices]  # points on the convex hull, in order

    x = hp[:, 0]
    y = hp[:, 1]

    # Fit circle: x^2 + y^2 = 2*a*x + 2*b*y + c  -->  solve for (a, b, c)
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    rhs = x**2 + y**2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = sol

    center = np.array([a, b])
    radius = np.sqrt(c + a**2 + b**2)
    return radius, center, hull

def compute_embeddings_panmictic(Ne, link, p, n=50, N_min = 1, N_max = 1e8, init_seed=42):
    np.random.seed(init_seed)
    X_init = np.random.rand(n, 2)
    R = np.ones((n,n)) * 1/Ne
    np.fill_diagonal(R, np.nan)

    X,_,_,_ = optimize_timepoint(X_init, R, W_t=None, N_min=N_min, N_max=N_max, p=p, link=link)
    X = X - np.mean(X, axis=0) # center the embeddings
    return X
# %%
Ne_list = np.arange(1e2,1e4,1e2).astype(int)
N_min = 1
N_max = 1e8

link_functions = [
    {"link": "inverse_distance", "p": 0.5},
    {"link": "inverse_distance", "p": 1},
    {"link": "inverse_distance", "p": 2},
    {"link": "inverse_distance", "p": 4},
    {"link": "exponential", "p": 0.5},
    {"link": "exponential", "p": 1},
    {"link": "exponential", "p": 2},
    {"link": "exponential", "p": 4}
]

radii = {}
average_pairwise_distances = {}
f_inverse = {}
for link_dict in link_functions:
    link = link_dict["link"]
    p = link_dict["p"]
    link_radii = []
    link_average_pairwise_distances = []
    link_f_inverse = []
    for Ne in tqdm(Ne_list):
        X = compute_embeddings_panmictic(Ne, link=link, p=p, N_min=N_min, N_max=N_max)
        D = squareform(pdist(X, metric='euclidean'))
        radius = hull_circle_radius(X)[0]
        link_average_pairwise_distances.append(np.mean(D[D!=0]))
        link_radii.append(radius)
        link_f_inverse.append(LinkFunction(name=link, N_min=N_min, p=p).inverse(1/Ne))
    radii[f"{link}_{p}"] = link_radii
    average_pairwise_distances[f"{link}_{p}"] = link_average_pairwise_distances
    f_inverse[f"{link}_{p}"] = link_f_inverse
# %%
fig, axs = plt.subplots(1, 2, figsize=(6, 3.5), sharex=True, sharey=False)
# panel layout: cols = link family, showing average pairwise distance only
panels = {
    "inverse_distance": axs[0],
    "exponential": axs[1],
}
# twin_axs = {key: ax.twinx() for key, ax in panels.items()}

for link in link_functions:
    p = link["p"]
    link_name = link["link"]
    key = f"{link_name}_{p}"
    ax = panels[link_name]
    # twin_ax = twin_axs[link_name]
    ax.plot(Ne_list, average_pairwise_distances[key], label=f"p={p}")
    # twin_ax.plot(Ne_list, f_inverse[key], '--', label=f"p={p}")

for link_name, ax in panels.items():
    # twin_ax = twin_axs[link_name]
    # ax.spines['right'].set_visible(True)
    ax.set_xlabel("Effective population size (Ne)")
    ax.set_ylabel("Average pairwise distance")
    # twin_ax.set_ylabel("f_inverse(1/Ne)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    # twin_ax.set_yscale('log')
axs[0].plot([1e2,1e4],[1e2,1e4], 'k--', label="y=x")
axs[0].set_title(r"$f(x) = (1 + x^p)^{-1}$")
axs[1].set_title(r"$f(x) = e^{-x^p}$")
axs[0].legend()
axs[1].legend()
fig.tight_layout()
file_name = os.path.splitext(os.path.basename(__file__))[0]
fig.savefig(f"../sim_figs/{file_name}.pdf", dpi=500)
# %%
