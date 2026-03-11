#%%
import numpy as np
import pandas as pd
import os
import tskit
import sys
sys.path.append("../../utils")
import matplotlib.pyplot as plt
from plotting_style import *
pop_colors = pd.read_csv("../metadata/DeManuel_pop_colors.csv")
pop_labels = pd.read_csv("../metadata/DeManuel_pop_labels.csv")
#%%
ts = tskit.load("../trees/DeManuel/chr21/chr21_0.trees")
pca = ts.pca(num_components=2).factors
# %%
fig,ax = plt.subplots(figsize=(6,4),dpi=500)
for pop in pop_colors["population"]:
    idx = pop_labels[pop_labels["population"]==pop].index
    ax.scatter(pca[idx,0],pca[idx,1],color=pop_colors[pop_colors["population"]==pop]["color"].values[0],label=pop,alpha=0.7)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig("DeManuel_pca.png")
# %%
df = pd.read_csv("../metadata/gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta_clean.csv")
# %%
df.project.unique()
# %%
