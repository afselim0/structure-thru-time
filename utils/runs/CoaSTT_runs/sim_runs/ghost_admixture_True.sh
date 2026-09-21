#!/bin/bash
# CoaSTT run: ghost_admixture, local ancestry inferred with True.
#
#     bash ghost_admixture_True.sh           # here
#     bash ghost_admixture_True.sh --slurm   # as a SLURM job
#
# One script per LAI method. Outputs are namespaced under
# ancestry_STT/True/, so the methods for a demography accumulate in the
# same run directory and share the upstream stages (tree simulation, coalescent
# pdf, genome-wide rates) with whichever method runs first.
#
# 'True' uses the ground-truth ancestry recorded from msprime census nodes at
# simulation time; it requires trees simulated by this pipeline, not external ones.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=12 mem_per_cpu=4G time=12:00:00

# LAI_method is passed on the command line rather than in the config because
# an unquoted True parses as a YAML boolean, not the string the pipeline wants.
coastt_run STT_sim --configfile ../data/configs/sim_configs/ghost_admixture.yaml \
    --config LAI_method=True
