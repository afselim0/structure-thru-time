#!/bin/bash
# CoaSTT run: recent_bottleneck
# Resource hints below are read by ../../generate_sim_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=12
# MEM_PER_CPU=4G
# TIME=24:00:00
set -euo pipefail

# All pipeline paths are relative to utils/, so run from there.
cd "$(dirname "$0")/../../.."

# This demography is run once per local-ancestry method. Outputs are namespaced
# under ancestry_STT/<method>/, so the runs accumulate in one output directory.
# (YAML parses an unquoted True as a boolean, hence --config LAI_method=True is
# passed on the command line rather than set in the config.)
for method in RFmix GNN; do
    echo "=== LAI_method=$method ==="
    snakemake STT_sim \
        --configfile ../data/configs/sim_configs/recent_bottleneck.yaml \
        --config LAI_method="$method" \
        --cores all --rerun-incomplete "$@"
done
