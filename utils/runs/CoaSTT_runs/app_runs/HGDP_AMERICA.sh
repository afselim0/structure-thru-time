#!/bin/bash
# CoaSTT run: the HGDP AMERICA sample set, from the SINGER-inferred tree sequences.
#
#     bash HGDP_AMERICA.sh           # here
#     bash HGDP_AMERICA.sh --slurm   # as a SLURM job
#
# 61 samples, 14 GB of tree sequences over 1,337 contigs -- one of the two small sets.
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

SAMPLE_SET=AMERICA
HUMAN_DATA="$COASTT_REPO/data/human_data"
LABELS="$HUMAN_DATA/resources/population_labels_colors"

# AMERICA: 16 workers at 16 cpus, 48G. Per-worker peak in compute_pdf is set by
# ts.pair_coalescence_counts: nodes x haplotypes x 8 bytes of per-node state
# (one sample set per haplotype) plus ~4.8 x pairs x 1202 x 8 bytes of output
# buffers. Contig size drifts along the sorted processing order, so size from
# the heaviest stretch, not the average.
# This set is small enough that both terms are a few GB per worker.
#
# num_threads comes from humans.yaml (16). It is restated on the command line
# because snakemake keeps only the LAST --config group: if run_common.sh appends
# its own num_threads group, the four paths below are silently discarded and the
# run dies on the empirical-mode pop_labels/pop_colors check. Passing it here
# suppresses that append.
NUM_THREADS=16

# 6 h: counting projects at 4.3 h or less for this set at 16 workers (scaled
# from the 3.8 contigs/min measured on ALL), so a short wall leaves margin and
# schedules sooner by backfill. The two largest sets ask for 12 h instead.
coastt_resources cpus=16 mem=48G time=06:00:00


coastt_run STT \
    --configfile ../data/configs/app_configs/humans.yaml \
    --config \
        "num_threads=$NUM_THREADS" \
        "output_path=$COASTT_REPO/results/outputs/HGDP/$SAMPLE_SET" \
        "trees_dir=$HUMAN_DATA/results/$SAMPLE_SET/ts_chunks" \
        "pop_labels=$LABELS/${SAMPLE_SET}_pop_labels.csv" \
        "pop_colors=$LABELS/${SAMPLE_SET}_pop_colors.csv"
