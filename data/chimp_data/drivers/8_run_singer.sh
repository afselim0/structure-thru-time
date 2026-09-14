#!/usr/bin/env bash
# STEP 8 -- infer the ARGs with SINGER, and date them with POLEGON.
#
#   bash drivers/8_run_singer.sh                          # all 23 autosomes
#   bash drivers/8_run_singer.sh --chrom chr21            # one chromosome
#   bash drivers/8_run_singer.sh --partition jnovembre    # dedicated partition instead
#   bash drivers/8_run_singer.sh -n                       # dry run: show what would be submitted
#
# Needs step 7. Writes results/chimp/singer-snakemake_outputs/<chrom>/trees/<chrom>.<rep>.tsz
#
# Submits ONE whole-node SLURM job per chromosome (caslake: 48 cores, 180 GB, exclusive)
# and returns immediately. Each job runs the whole workflow for its chromosome with
# `snakemake --cores all` and no executor, which is how
# singer-snakemake is designed to be run -- see the note in scripts/run_singer.sh for why
# per-rule SLURM submission is the wrong granularity for `run_polegon` (17,940 jobs of a few
# seconds each).
#
# Monitor with `squeue -u $USER` and `tail -f logs/chimp_singer-*.out`. Expect roughly a day.
#
# Scientific parameters come from drivers/singer_config.yaml. Steps are resumable: a
# chromosome whose job is killed can simply be resubmitted, and the `.singer.done` sentinels
# mean finished chunks are not recomputed.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "${ROOT}/scripts/run_singer.sh" "$@"
