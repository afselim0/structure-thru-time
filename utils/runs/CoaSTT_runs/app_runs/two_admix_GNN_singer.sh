#!/bin/bash
# CoaSTT run: two_admix with GNN local ancestry, on trees inferred with SINGER
# (+POLEGON).
#
#     bash two_admix_GNN_singer.sh           # here
#     bash two_admix_GNN_singer.sh --slurm   # as a SLURM job
#
# Runs the same pipeline as sim_runs/two_admix_GNN.sh, but on inferred rather
# than simulated trees, writing to a sibling "singer" directory beside that
# run's "sim" directory. Run sim_runs/two_admix_GNN.sh first: its simulated
# trees are this run's input.
#
# GNN (not True) is the LAI method here: ground-truth ancestry comes from the
# msprime census nodes recorded at simulation time, and the snakefile rejects
# LAI_method='True' whenever the trees are not the ones this run simulated.
# Inferred trees therefore require an inference-based LAI method.
#
# Sizing: SINGER (-n 50) costs ~13 min per 5 Mb contig per worker here, so 500
# contigs on 48 workers take ~3 h of inference; ~0.5 GB per worker. Under
# --slurm the pipeline's workers are matched to the allocated CPUs.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=48 mem_per_cpu=2G time=10:00:00

coastt_run_inferred two_admix singer LAI_method=GNN
