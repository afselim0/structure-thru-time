#!/bin/bash
# CoaSTT run: the HGDP EAST_ASIA sample set, from the SINGER-inferred tree sequences.
#
#     bash HGDP_EAST_ASIA.sh           # here
#     bash HGDP_EAST_ASIA.sh --slurm   # as a SLURM job
#
# 223 samples, 37 GB of tree sequences over 1,344 contigs -- the most samples of any set, so the most population pairs to embed.
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

SAMPLE_SET=EAST_ASIA
HUMAN_DATA="$COASTT_REPO/data/human_data"
LABELS="$HUMAN_DATA/resources/population_labels_colors"

# EAST_ASIA: 16 workers at 16 cpus, 128G. Per-worker peak in compute_pdf is set by
# ts.pair_coalescence_counts: nodes x haplotypes x 8 bytes of per-node state
# (one sample set per haplotype) plus ~4.8 x pairs x 1202 x 8 bytes of output
# buffers. Contig size drifts along the sorted processing order, so size from
# the heaviest stretch, not the average.
# For this set the 16 heaviest consecutive contigs sum to ~99 GB, ~81 GB
# on a typical stretch.
#
# num_threads comes from humans.yaml (16). It is restated on the command line
# because snakemake keeps only the LAST --config group: if run_common.sh appends
# its own num_threads group, the four paths below are silently discarded and the
# run dies on the empirical-mode pop_labels/pop_colors check. Passing it here
# suppresses that append.
NUM_THREADS=16

# 12 h rather than the 6 h the smaller sets get: this is one of the two largest
# sets here, and counting alone projects past 6 h at 16 workers (scaled from the
# 3.8 contigs/min measured on ALL). Counting is resumable, so a timeout would
# cost only the contigs in flight -- but a wall long enough to finish in one go
# is worth more than the marginal backfill advantage of a shorter one.
coastt_resources cpus=16 mem=128G time=12:00:00


coastt_run STT \
    --configfile ../data/configs/app_configs/humans.yaml \
    --config \
        "num_threads=$NUM_THREADS" \
        "output_path=$COASTT_REPO/results/outputs/HGDP/$SAMPLE_SET" \
        "trees_dir=$HUMAN_DATA/results/$SAMPLE_SET/ts_chunks" \
        "pop_labels=$LABELS/${SAMPLE_SET}_pop_labels.csv" \
        "pop_colors=$LABELS/${SAMPLE_SET}_pop_colors.csv"
