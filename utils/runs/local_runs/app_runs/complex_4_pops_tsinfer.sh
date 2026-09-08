#!/bin/bash
# CoaSTT run: complex_4_pops, on trees inferred with tsinfer.
#
# Runs the same pipeline as the matching utils/runs/local_runs/sim_runs script,
# but on inferred rather than simulated trees, writing to a sibling "tsinfer"
# directory beside that run's "sim" directory. Run the sim_runs script first:
# the VCFs it writes are this run's input.
#
# Resource hints below are read by ../../generate_app_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=12
# MEM_PER_CPU=4G
# TIME=12:00:00
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# All pipeline paths are relative to utils/, so run from there.
cd "$script_dir/../../.."

# Extra flags given to this script are forwarded to snakemake (e.g. -n).
SNAKEMAKE_ARGS=("$@")
source "$script_dir/../run_inferred.sh"

run_inferred complex_4_pops tsinfer
