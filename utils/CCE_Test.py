#%%
import os
import sys
sys.path.append("../src")
from CountCoalEvents import *
#%%
ts_dir = "../data/trees/DeManuel/all_chopped"
ts_files = os.listdir(ts_dir)
max_time = 2000
num_posterior_samples = 100
num_timepoints = 1000
# %%
CCE = CountCoalEvents(ts_dir, max_time = max_time, num_posterior_samples = num_posterior_samples, num_timepoints = num_timepoints)
#%%
CCE.max_time = 100000
CCE.num_timepoints = 5000
x = CCE.count_coal_segments(CCE.ts_files[0])
# %%
