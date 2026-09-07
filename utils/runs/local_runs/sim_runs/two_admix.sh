#!/bin/bash
# CoaSTT run: two_admix
# Resource hints below are read by ../../generate_sim_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=12
# MEM_PER_CPU=4G
# TIME=24:00:00
set -euo pipefail

# All pipeline paths are relative to utils/, so run from there.
cd "$(dirname "$0")/../../.."

snakemake STT_sim \
    --configfile ../data/configs/sim_configs/two_admix.yaml \
    --cores all --rerun-incomplete "$@"
