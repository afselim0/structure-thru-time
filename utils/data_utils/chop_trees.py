import glob
import tskit
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def chop_ts(ts,contig_length,trim=1e6):
    total_span = ts.sequence_length
    if ts.first().total_branch_length == 0:
        start_pos = ts.first().span + trim
        print(f"Warning: first tree has zero branch length, starting at {ts.first().span} instead of {0}")
    else:
        start_pos = trim
    end_pos = total_span - trim
    breakpoints = np.arange(start_pos,end_pos+1,contig_length).astype(int)
    for i in range(len(breakpoints)-1):
        start = breakpoints[i]
        end = breakpoints[i+1]
        chopped_ts = ts.keep_intervals([(start, end)],simplify=True).trim()
        yield chopped_ts, start, end


def main():
    parser = ArgumentParser(description="Chop chimp trees into smaller segments")
    parser.add_argument("--ts_dir", type=str, required=True, help="Directory containing the .trees files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the chopped trees")
    parser.add_argument("--contig_length", type=int, default=5e6, help="Length of each contig")
    parser.add_argument("--trim", type=int, default=1e6, help="Length to trim from the start and end of each tree sequence")
    args = parser.parse_args()
    ts_dir = args.ts_dir
    chopped_trees_path = args.output_dir
    contig_length = args.contig_length
    trim = args.trim

    os.makedirs(chopped_trees_path, exist_ok=True)
    ts_files = glob.glob(os.path.join(ts_dir, "*.trees"))
    for ts_file in tqdm(ts_files):
        print(f"Processing file: {ts_file}")
        ts = tskit.load(ts_file)
        num_chunks = int(np.ceil(ts.sequence_length / contig_length))
        contig_id = os.path.basename(ts_file).removesuffix(".trees")
        for i, (chopped_ts, start, end) in enumerate(chop_ts(ts, contig_length, trim)):
            out_file = f"{contig_id}_{int(start)}_{int(end)}.trees"
            chopped_ts.dump(os.path.join(chopped_trees_path, out_file))

if __name__ == "__main__":
    main()

#%%
# contig_names_dict = {
#     "CM009238.2": "chr1",
#     "CM009239.2": "chr2A",
#     "CM009240.2": "chr2B",
#     "CM009241.2": "chr3",
#     "CM009242.2": "chr4",
#     "CM009243.2": "chr5",
#     "CM009244.2": "chr6",
#     "CM009245.2": "chr7",
#     "CM009246.2": "chr8",
#     "CM009247.2": "chr9",
#     "CM009248.2": "chr10",
#     "CM009249.2": "chr11",
#     "CM009250.2": "chr12",
#     "CM009251.2": "chr13",
#     "CM009252.2": "chr14",
#     "CM009253.2": "chr15",
#     "CM009254.2": "chr16",
#     "CM009255.2": "chr17",
#     "CM009256.2": "chr18",
#     "CM009257.2": "chr19",
#     "CM009258.2": "chr20",
#     "CM009259.2": "chr21",
#     "CM009260.2": "chr22",
# }
# # for ts_file in ts_files:
# for k in range(1171, len(ts_files)):
#     ts_file = ts_files[k]
#     print(f"Processing file {k+1}/{len(ts_files)}: {ts_file}")
#     ts = tskit.load(ts_file)
#     num_chunks = int(np.ceil(ts.sequence_length / contig_length))
#     contig_id = ts_file[224:234]
#     chr_number = contig_names_dict[contig_id]
#     replicate_id = os.path.basename(ts_file).removesuffix(".trees").split("_")[-1]
#     for i, (chopped_ts, start, end) in tqdm(enumerate(chop_ts(ts,contig_length)),total=num_chunks):
#         out_file = f"{chr_number}_{int(start)}_{int(end)}_{replicate_id}.trees"
#         chopped_ts.dump(os.path.join(chopped_trees_path,out_file))
#     # k +=1
# # %%
