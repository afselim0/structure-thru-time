#!/bin/bash
# CoaSTT run: three_way_admixture, local ancestry inferred with GNN.
#
# One script per LAI method. Outputs are namespaced under
# ancestry_STT/GNN/, so the methods for a demography accumulate in the
# same run directory and share the upstream stages (tree simulation, coalescent
# pdf, genome-wide rates) with whichever method runs first.
#
# Resource hints below are read by ../../generate_sim_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=12
# MEM_PER_CPU=4G
# TIME=24:00:00
set -euo pipefail

# All pipeline paths are relative to utils/, so run from there.
cd "$(dirname "$0")/../../.."

# LAI_method is passed on the command line because an unquoted True parses as a
# YAML boolean rather than the string the pipeline expects.
snakemake STT_sim \
    --configfile ../data/configs/sim_configs/three_way_admixture.yaml \
    --config LAI_method=GNN \
    --cores all --rerun-incomplete "$@"
