#%%
import glob
import tskit
import os
import numpy as np
from tqdm import tqdm
#%%
def chop_ts(ts,contig_length,remove_first = False):
    if remove_first:
        first_tree_span = ts.first().span
    else:
        first_tree_span = 0
    total_span = ts.sequence_length
    breakpoints = np.arange(first_tree_span,total_span+1,contig_length)
    for i in range(len(breakpoints)-1):
        start = breakpoints[i]
        end = breakpoints[i+1]
        chopped_ts = ts.keep_intervals([(start, end)],simplify=True).trim()
        yield chopped_ts,start,end

#%%
ts_files = glob.glob("./DeManuel/chr21/*.trees")
chopped_trees_path = "DeManuel/chr21_chopped"
contig_length = 1e6
for ts_file in tqdm(ts_files,total=len(ts_files)):
    ts = tskit.load(ts_file)
    replicate_id = os.path.basename(ts_file).removesuffix(".trees").split("_")[1]
    for i, (chopped_ts, start, end) in enumerate(chop_ts(ts,contig_length)):
        out_file = f"chr21_{int(start)}_{int(end)}_{replicate_id}.trees"
        chopped_ts.dump(os.path.join(chopped_trees_path,out_file))
# %%
