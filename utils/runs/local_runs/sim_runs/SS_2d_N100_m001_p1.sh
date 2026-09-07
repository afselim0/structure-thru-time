#!/bin/bash
# CoaSTT run: SS_2d_N100_m001_p1
# Resource hints below are read by ../../generate_sim_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=24
# MEM_PER_CPU=4G
# TIME=08:00:00
set -euo pipefail

# All pipeline paths are relative to utils/, so run from there.
cd "$(dirname "$0")/../../.."

snakemake STT_sim \
    --configfile ../data/configs/sim_configs/SS_2d_N100_m001_p1.yaml \
    --cores all --rerun-incomplete "$@"
