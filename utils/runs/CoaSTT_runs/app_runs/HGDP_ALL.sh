#!/bin/bash
# CoaSTT run: the HGDP ALL sample set, from the SINGER-inferred tree sequences.
#
#     bash HGDP_ALL.sh           # here
#     bash HGDP_ALL.sh --slurm   # as a SLURM job
#
# 216 samples (4 per population from each of 54 populations), 61 GB of tree sequences over 1,344 contigs -- the largest set, and the only one labelled by region rather than by population.
#
# All eight HGDP sets share one config, data/configs/app_configs/humans.yaml;
# only the four per-set paths below differ, so they are passed with --config
# rather than by copying the config eight times. The sets write to separate
# output_paths and read separate trees_dirs, so they share no files and can all
# run at once.
#
# num_threads goes in the same --config group as the four paths: snakemake
# keeps only the LAST --config group, so letting run_common.sh append its own
# would silently drop the paths and the run would die on the empirical-mode
# check for pop_labels/pop_colors. run_common.sh appends nothing when it sees a
# num_threads= already among the arguments, so the value is set here instead --
# from the allocation under --slurm, from the cpuset when running locally.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

SAMPLE_SET=ALL
HUMAN_DATA="$COASTT_REPO/data/human_data"
LABELS="$HUMAN_DATA/resources/population_labels_colors"

# 16 cpus, 170 GB, 12 counting workers -- fits one caslake node (48 cpus, 180 GB).
# Per-worker peak in compute_pdf is set by ts.pair_coalescence_counts:
#   nodes x haplotypes x 8 bytes   (per-node working state, one sample set per
#                                   haplotype; ~5.7 GB on an average ALL contig)
# + ~4.8 x pairs x 1202 x 8 bytes  (output buffers; ~4.3 GB for ALL's 93,096 pairs)
# i.e. 10 GB on an average contig and 11.9 GB on the largest. Contig size drifts
# along the sorted processing order, and the chr2-chr5 stretch is the heaviest;
# a 32-worker run was OOM-killed at 236 GB on entering it. Twelve workers on the
# worst run of consecutive contigs sum to 143 GB, 120 GB typically.
coastt_resources cpus=16 mem=170G time=16:00:00

NUM_THREADS=12
_avail="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 20)}"
[[ $COASTT_SLURM -eq 0 && $_avail -lt $NUM_THREADS ]] && NUM_THREADS=$_avail

coastt_run STT \
    --configfile ../data/configs/app_configs/humans.yaml \
    --config \
        "num_threads=$NUM_THREADS" \
        "output_path=$COASTT_REPO/results/outputs/HGDP/$SAMPLE_SET" \
        "trees_dir=$HUMAN_DATA/results/$SAMPLE_SET/ts_chunks" \
        "pop_labels=$LABELS/${SAMPLE_SET}_pop_labels.csv" \
        "pop_colors=$LABELS/${SAMPLE_SET}_pop_colors.csv"
