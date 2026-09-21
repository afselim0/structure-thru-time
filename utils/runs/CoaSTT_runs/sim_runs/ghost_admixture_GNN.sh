#!/bin/bash
# CoaSTT run: ghost_admixture, local ancestry inferred with GNN.
#
#     bash ghost_admixture_GNN.sh           # here
#     bash ghost_admixture_GNN.sh --slurm   # as a SLURM job
#
# One script per LAI method. Outputs are namespaced under
# ancestry_STT/GNN/, so the methods for a demography accumulate in the
# same run directory and share the upstream stages (tree simulation, coalescent
# pdf, genome-wide rates) with whichever method runs first.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=12 mem_per_cpu=4G time=24:00:00

# LAI_method is passed on the command line to match the True variant, where an
# unquoted True would parse as a YAML boolean rather than the expected string.
coastt_run STT_sim --configfile ../data/configs/sim_configs/ghost_admixture.yaml \
    --config LAI_method=GNN
