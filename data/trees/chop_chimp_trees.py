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
ts_dir = "/project/jnovembre/data/external_private/chimp_data/scratch_isilon/groups/compgen/nhermosilla/PROJECTS/Structure_thru_time/pantro_data/SINGER/parallel_SINGER/pan_pbl"
ts_files = glob.glob(os.path.join(ts_dir,"*.trees"))
chopped_trees_path = "DeManuel/all_chopped"
contig_length = 5e6

contig_names_dict = {
    "CM009238.2": "chr1",
    "CM009239.2": "chr2A",
    "CM009240.2": "chr2B",
    "CM009241.2": "chr3",
    "CM009242.2": "chr4",
    "CM009243.2": "chr5",
    "CM009244.2": "chr6",
    "CM009245.2": "chr7",
    "CM009246.2": "chr8",
    "CM009247.2": "chr9",
    "CM009248.2": "chr10",
    "CM009249.2": "chr11",
    "CM009250.2": "chr12",
    "CM009251.2": "chr13",
    "CM009252.2": "chr14",
    "CM009253.2": "chr15",
    "CM009254.2": "chr16",
    "CM009255.2": "chr17",
    "CM009256.2": "chr18",
    "CM009257.2": "chr19",
    "CM009258.2": "chr20",
    "CM009259.2": "chr21",
    "CM009260.2": "chr22",
}
#%%
# k=0
# for ts_file in ts_files:
for k in range(1171, len(ts_files)):
    ts_file = ts_files[k]
    print(f"Processing file {k+1}/{len(ts_files)}: {ts_file}")
    ts = tskit.load(ts_file)
    num_chunks = int(np.ceil(ts.sequence_length / contig_length))
    contig_id = ts_file[224:234]
    chr_number = contig_names_dict[contig_id]
    replicate_id = os.path.basename(ts_file).removesuffix(".trees").split("_")[-1]
    for i, (chopped_ts, start, end) in tqdm(enumerate(chop_ts(ts,contig_length)),total=num_chunks):
        out_file = f"{chr_number}_{int(start)}_{int(end)}_{replicate_id}.trees"
        chopped_ts.dump(os.path.join(chopped_trees_path,out_file))
    # k +=1
# %%
