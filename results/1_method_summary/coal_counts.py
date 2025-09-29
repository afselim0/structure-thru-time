#%%
import msprime
import numpy as np
import sys
import matplotlib.cm
import matplotlib.colors
sys.path.append("../../../src/")
from counting import *
from latent_embeddings import *
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'  # Generic sans-serif
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Tahoma']  # Fallback order

def get_edge_path(tree,sample,node):
    path = []
    current_node = sample
    while current_node != node:
        path.append(current_node)
        current_node = tree.parent(current_node)

    return path

def get_active_edge_nodes(tree,active_pair):
    active_edge_nodes = []
    i,j = active_pair[0], active_pair[1]
    mrca = tree.mrca(i,j)
    path_i = get_edge_path(tree,i,mrca)
    path_j = get_edge_path(tree,j,mrca)
    active_edge_nodes.extend(path_i)
    active_edge_nodes.extend(path_j)
        
    return np.unique(active_edge_nodes)

def pop_split(N0,N1,N2,t_split):
    demography = msprime.Demography()
    demography.add_population(name="main", initial_size= N0)

    demography.add_population(name="A", initial_size= N1, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= N2, default_sampling_time = 0)

    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="main")
    
    return demography

N0 = 10000
N1 = 10000
N2 = 10000

demography = pop_split(N0,N1,N2,t_split=2000)
# %%
# Tree subfigure
n = 10
ts = msprime.sim_ancestry(samples = {"A":n,"B":n},
                          demography = demography,
                          random_seed=42,
                          ploidy=1)
n_samples = ts.num_samples
tree = ts.first()

pairs = np.column_stack(np.triu_indices(n_samples, k=1)) 
num_pairs = pairs.shape[0]

np.random.seed(5)
active_pairs = [
    [4,8],
    [3,19],
    [0,6],
    [2,12]
]
num_colors = len(active_pairs)
color_values = np.linspace(0, 1, num_colors)
viridis = matplotlib.cm.get_cmap("viridis")
viridis_colors = [matplotlib.colors.rgb2hex(viridis(val)) for val in color_values]

styles = []
for i in list(tree.nodes()):  # Assuming `n_samples` is 50 (or more)
    if i < n:
        c = "red"
    elif i < 2*n:
        c = "blue"
    else:
        c = None
    styles.append(f".node.n{i} > .sym {{fill: {c}}}")  # Correct CSS selector

y_ticks = [2000]
for color,pair in zip(viridis_colors,active_pairs):
    edge_nodes = get_active_edge_nodes(tree,pair)
    mrca = tree.mrca(pair[0], pair[1])
    styles.append(f".node.n{mrca} > .sym {{fill: {color}}}")
    for node in edge_nodes:
        styles.append(f".n{node} > .edge {{stroke: {color}; stroke-width: 2px}}")

style_string = "\n".join(styles)

css_string = ".tree .lab {font-family: sans-serif}" + ".y-axis .grid {stroke: black ; stroke-width: 2px}" + style_string
tree.draw_svg("tree.svg",
    size=(400,400),
    node_labels = {},
    style=css_string)
    # y_axis=True,
    # y_gridlines=True,
    # y_ticks=y_ticks)

#%%
ts_list = msprime.sim_ancestry(
    samples = {"A":n,"B":n},
    demography = demography,
    random_seed=42,
    sequence_length=1,
    ploidy = 1,
    num_replicates = 200000
    )
trees = [ts.first() for ts in ts_list]
delta = 100
times = np.array([0,5000])
N, L = count_coal_events(trees,times, delta)
#%%
N0,N2000 = N[:,:,0], N[:,:,1]
L0,L2000 = L[:,:,0], L[:,:,1]
N0 = np.triu(N0) + np.triu(N0, k=1).T
N2000 = np.triu(N2000) + np.triu(N2000, k=1).T
L0 = np.triu(L0) + np.triu(L0, k=1).T
L2000 = np.triu(L2000) + np.triu(L2000, k=1).T
#%%
plt.rcParams.update({
    'font.size': 8,       
    'axes.titlesize': 8,  
    'axes.labelsize': 6,  
    'xtick.labelsize': 6,  
    'ytick.labelsize': 6,  
    'legend.fontsize': 6,  
})

fig,ax = plt.subplots(2,2, figsize=(4.17,4.17),dpi=500)

heatmap0 = sns.heatmap(N0, vmin=0, vmax=250,cbar=True,xticklabels=False, yticklabels=False, ax=ax[1,0])
heatmap1 = sns.heatmap(N2000, vmin=0, vmax=250,cbar=True,xticklabels=False, yticklabels=False, ax=ax[0,0])
heatmap2 = sns.heatmap(L0, vmin=0, vmax=20000,cbar=True,xticklabels=False, yticklabels=False, ax=ax[1,1], cmap="mako")
heatmap3 = sns.heatmap(L2000, vmin=0, vmax=20000,cbar=True,xticklabels=False, yticklabels=False, ax=ax[0,1], cmap="mako")

cbar = heatmap0.collections[0].colorbar
cbar.set_ticks([0, 250])  # Override the ticks

cbar = heatmap2.collections[0].colorbar
cbar.set_ticks([0, 20000])  # Override the ticks

cbar = heatmap1.collections[0].colorbar
cbar.set_ticks([0, 250])  # Override the ticks

cbar = heatmap3.collections[0].colorbar
cbar.set_ticks([0, 20000])  # Override the ticks

ax[0,0].set_title("N(t >> split time)")
ax[0,1].set_title("L(t >> split time)")
ax[1,0].set_title("N(t << split time)")
ax[1,1].set_title("L(t << split time)")

fig.savefig("coal_counts.png", dpi=500)
# %%
alpha = 0.05
R0 = N0 / L0
R0[np.arange(R0.shape[0]), np.arange(R0.shape[1])] = 0
R2000 = N2000 / L2000
R2000[np.arange(R2000.shape[0]), np.arange(R2000.shape[1])] = 0

np.random.seed(104)
X_init = np.random.rand(R0.shape[0], 2)
X0, _ = bfgs_optimization(X_init, R0, alpha)
X2000, _ = bfgs_optimization(X_init, R2000, alpha)
X0 = X0 - np.mean(X0, axis=0)
X2000 = X2000 - np.mean(X2000, axis=0)

fig,ax = plt.subplots(2,1,figsize=(4.6/2,4.5),dpi=500,sharex=True,sharey=True)

ax[1].scatter(X0[:10, 0], X0[:10, 1], c='red', s=5)
ax[1].scatter(X0[10:, 0], X0[10:, 1], c='blue', s=5)


ax[0].scatter(X2000[:10, 0], X2000[:10, 1], c='red', s=5, label='A')
ax[0].scatter(X2000[10:, 0], X2000[10:, 1], c='blue', s=5, label='B')
ax[0].legend(loc='upper right', fontsize=6, markerscale=0.5, frameon=False)


for a in ax:
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['left'].set_linewidth(0.5)
    a.spines['bottom'].set_linewidth(0.5)
    a.tick_params(width=0.5)
    a.set_xlabel("X1")
    a.set_ylabel("X2")   
    a.set_xlim(-3.2,3.2)
    a.set_ylim(-3.2, 3.2)   
    a.set_xticks([-2,0,2])
    a.set_yticks([-2,0,2])

fig.savefig("embeddings.png", dpi=500)
# %%
