#%%
import tskit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
ts = tskit.load("plane_test.trees")
# slimd_ids = np.floor(np.array([ts.node(i).metadata["slim_id"] for i in range(ts.num_samples)])/2).astype(int)
# locations_df = pd.read_csv('locations', sep='\t')
# locations = locations_df[locations_df['ind'].isin(slimd_ids)][["x", "y"]].values
# %%
locations_df = pd.read_csv('locations', sep='\t')
init_locs = locations_df[locations_df['gen']==1000][["x", "y"]].values
final_locs = locations_df[locations_df['gen']==0][["x", "y"]].values
plt.scatter(final_locs[:, 0], final_locs[:, 1], c='red')
plt.scatter(init_locs[:, 0], init_locs[:, 1], s=1, c='black',alpha=0.2)
plt.xlim(0, 1000)
plt.ylim(0, 1000)
