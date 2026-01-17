#%%
import glob
import tskit
import os
import numpy as np
from tqdm import tqdm
#%%
chopped_trees_path = "../wohns_chopped"
chopped_vcf_path = "../wohns_vcfs"

ts_files = glob.glob("../wohns/*.trees")
contig_length = 5e6
#%%
for ts_file in ts_files:
    ts = tskit.load(ts_file)
    first_tree_span = ts.first().span
    total_span = ts.sequence_length
    breakpoints = np.arange(first_tree_span,total_span,contig_length).astype(int)
    chromsome_num = os.path.basename(ts_file)[:-7]
    for i in tqdm(range(len(breakpoints)-1)):
        start = breakpoints[i]
        end = breakpoints[i+1]
        chopped_ts = ts.keep_intervals([(start, end)],simplify=True).trim()
        # Save files
        out_file = f"{chromsome_num}_{start}_{end}.trees"
        chopped_ts.dump(os.path.join(chopped_trees_path,out_file))

        # vcf_file = f"{chromsome_num}_{start}_{end}.vcf"
        # with open(os.path.join(chopped_vcf_path,vcf_file), "w") as vcf_output:
        #     chopped_ts.write_vcf(vcf_output, contig_id=f"chr{chromsome_num}",position_transform=lambda x: x + start)
# %%