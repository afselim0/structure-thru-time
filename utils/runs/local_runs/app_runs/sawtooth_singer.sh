#!/bin/bash
# CoaSTT run: sawtooth, on trees inferred with SINGER (+POLEGON).
#
# Runs the same pipeline as the matching utils/runs/local_runs/sim_runs script,
# but on inferred rather than simulated trees, writing to a sibling "singer"
# directory beside that run's "sim" directory. Run the sim_runs script first:
# its simulated trees are this run's input.
#
# Resource hints below are read by ../../generate_app_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=12
# MEM_PER_CPU=4G
# TIME=24:00:00
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# All pipeline paths are relative to utils/, so run from there.
cd "$script_dir/../../.."

# Extra flags given to this script are forwarded to snakemake (e.g. -n); SINGER at 500 contigs needs ~10 min per contig per core.
SNAKEMAKE_ARGS=("$@")
source "$script_dir/../run_inferred.sh"

run_inferred sawtooth singer
