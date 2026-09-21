#!/bin/bash
# CPUS=16
# MEM_PER_CPU=8G
# TIME=36:00:00
#
# The admixed chimp run: local ancestry for the NC and "W x C" individuals against two
# size-matched reference panels (EasternCentral = 5 Eastern + 5 Central individuals,
# West = 10 Western; 20 haplotypes each), then the whole rates/embeddings/plotting
# pipeline once genome-wide and once per ancestry panel.
#
# Sized like chimps.sh, which starts from the same 1,334 tree sequences (73 GB) and is
# dominated by the same compute_pdf counting step. Two differences push the wall clock
# up rather than down, so the 36 h is kept:
#   * infer_local_ancestry (GNN) walks every local tree of every contig before any
#     counting starts;
#   * compute_pdf then writes a per-ancestry pdf for each panel on top of the
#     genome-wide one, i.e. three counting passes rather than one.
# The counting is resumable -- an interrupted run validates and reuses the .npy files
# already written -- so a timeout costs only the contigs in flight.
#
# This run writes to results/outputs/chimps/admix and builds its own pdf
# (pdf_maxT200000), so it shares nothing with the chimps/no_admix runs and can go
# alongside them.
set -euo pipefail

# All pipeline paths are relative to utils/, so run from there.
cd "$(dirname "$0")/../../.."

# The system /lib64/libstdc++ (EL8) is older than numpy 2.5 needs, so snakemake
# dies with `GLIBCXX_3.4.29 not found` as soon as it imports pandas. Put the
# env's own lib first. Needed on login and compute nodes alike.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Match the compute_pdf workers to the CPUs actually available. SLURM_CPUS_PER_TASK
# is only set for the batch step, not for a plain `srun --overlap` step, so fall back
# to the cpuset size rather than a fixed default (which silently under-subscribed a
# 16-CPU allocation). With neither, leave the config's own num_threads alone.
cpus="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null)}"
snakemake STT \
    --configfile ../data/configs/app_configs/chimps_admix.yaml \
    ${cpus:+--config num_threads=$cpus} \
    --cores all --rerun-incomplete "$@"
