#!/bin/bash
# CoaSTT run: double_2d_SS, on trees inferred with SINGER (+POLEGON).
#
#     bash double_2d_SS_singer.sh           # here
#     bash double_2d_SS_singer.sh --slurm   # as a SLURM job
#
# Runs the same pipeline as the matching sim_runs script, but on inferred
# rather than simulated trees, writing to a sibling "singer" directory beside
# that run's "sim" directory. Run sim_runs/double_2d_SS.sh first: its
# simulated trees are this run's input.
#
# Sizing: SINGER (-n 50) costs ~11 min per 5 Mb contig per worker here, so 500
# contigs on 48 workers take ~2 h of inference; ~0.5 GB per worker. Under
# --slurm the pipeline's workers are matched to the allocated CPUs.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=48 mem_per_cpu=2G time=08:00:00

coastt_run_inferred double_2d_SS singer
