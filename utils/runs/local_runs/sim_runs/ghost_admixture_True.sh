#!/bin/bash
# CoaSTT run: ghost_admixture, local ancestry inferred with True.
#
# One script per LAI method. Outputs are namespaced under
# ancestry_STT/True/, so the methods for a demography accumulate in the
# same run directory and share the upstream stages (tree simulation, coalescent
# pdf, genome-wide rates) with whichever method runs first.
#
# 'True' uses the ground-truth ancestry recorded from msprime census nodes at
# simulation time; it requires trees simulated by this pipeline, not external ones.
# Resource hints below are read by ../../generate_sim_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=12
# MEM_PER_CPU=4G
# TIME=12:00:00
set -euo pipefail

# All pipeline paths are relative to utils/, so run from there.
cd "$(dirname "$0")/../../.."

# LAI_method is passed on the command line because an unquoted True parses as a
# YAML boolean rather than the string the pipeline expects.
snakemake STT_sim \
    --configfile ../data/configs/sim_configs/ghost_admixture.yaml \
    --config LAI_method=True \
    --cores all --rerun-incomplete "$@"
