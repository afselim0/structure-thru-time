#%%
import numpy as np
import pandas as pd
import os
import tskit
import sys
sys.path.append("../../utils")
import matplotlib.pyplot as plt
from plotting_style import *
pop_colors = pd.read_csv("../metadata/wohns_pop_colors.csv",index_col=0)
pop_labels = pd.read_csv("../metadata/wohns_pop_labels.csv",index_col=0)
active_samples = []
active_sample_colors = []
populations = pop_colors["population"].unique()
for pop in populations:
    idx = pop_labels[pop_labels["population"]==pop].index.values
    active_samples.extend(idx[:2])
    active_sample_colors.extend([pop_colors[pop_colors["population"]==pop]["color"].values[0]] * 2)

#%%
ts = tskit.load("../trees/wohns/chr1p.trees")
ts = ts.simplify(active_samples)
pca = ts.pca(num_components=2).factors
# %%
fig,ax = plt.subplots(figsize=(6,4),dpi=500)
ax.scatter(pca[:,0],pca[:,1],color=active_sample_colors,alpha=0.7)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig("wohns_pca.png")
# %%
