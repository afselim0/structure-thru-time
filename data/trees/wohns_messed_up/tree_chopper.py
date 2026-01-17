#%%
import glob
import tskit
import os
import numpy as np
from tqdm import tqdm
#%%
def chop_ts(ts,contig_length):
    first_tree_span = ts.first().span
    total_span = ts.sequence_length
    breakpoints = np.arange(first_tree_span,total_span,contig_length)
    for i in tqdm(range(len(breakpoints)-1)):
        start = breakpoints[i]
        end = breakpoints[i+1]
        chopped_ts = ts.keep_intervals([(start, end)],simplify=True).trim()
        yield chopped_ts

#%%
chopped_trees_path = "../wohns_chopped"

ts_files = glob.glob("../wohns/*.trees")
contig_length = 5e6
for ts_file in ts_files:
    ts = tskit.load(ts_file)
    for i, chopped_ts in enumerate(chop_ts(ts,contig_length)):
        out_file = f"{os.path.splitext(ts_file)[0]}_contig_{i}.trees"
        chopped_ts.dump(os.path.join(chopped_trees_path,out_file))
# %%
