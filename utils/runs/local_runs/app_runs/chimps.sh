#!/bin/bash
# CPUS=16
# MEM_PER_CPU=8G
# TIME=36:00:00
#
# Sized for a run that starts from the tree sequences.  compute_pdf dominates: it counts
# coalescent spans over 1,334 contigs (73 GB of .trees) into 1,431 pair x 10,000 timepoint
# arrays, ~115 MB per contig and ~153 GB of intermediates in counts_dir.  The counting is
# resumable -- an interrupted run validates and reuses the .npy files already written -- so
# a timeout costs only the contigs in flight, but the first pass needs the long wall clock.
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
    --configfile ../data/configs/app_configs/chimps.yaml \
    ${cpus:+--config num_threads=$cpus} \
    --cores all --rerun-incomplete "$@"
