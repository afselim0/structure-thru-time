#!/bin/bash
# CoaSTT run: split_sawtooth, on trees inferred with SINGER (+POLEGON).
#
# Runs the same pipeline as the matching utils/runs/local_runs/sim_runs script,
# but on inferred rather than simulated trees, writing to a sibling "singer"
# directory beside that run's "sim" directory. Run the sim_runs script first:
# its simulated trees are this run's input.
#
# Resource hints below are read by ../../generate_app_results_slurm.sh when it
# builds the job array; they are ignored by the local driver.
# CPUS=48
# MEM_PER_CPU=2G
# TIME=12:00:00
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

# All pipeline paths are relative to utils/, so run from there.
cd "$script_dir/../../.."

# Extra flags given to this script are forwarded to snakemake (e.g. -n).
# Sizing: SINGER (-n 50) costs ~25 min per 5 Mb contig per worker here, so 500 contigs
# on CPUS workers take ~5 h of inference; ~0.5 GB per worker. The
# config's num_threads is overridden below so the workers match the allocated CPUs.
SNAKEMAKE_ARGS=("$@")
source "$script_dir/../run_inferred.sh"

run_inferred split_sawtooth singer num_threads=${SLURM_CPUS_PER_TASK:-12}
