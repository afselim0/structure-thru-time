#!/bin/bash
# CoaSTT run: two_admix with GNN local ancestry, on trees inferred with tsinfer.
#
# Runs the same pipeline as sim_runs/two_admix_GNN.sh, but on inferred rather
# than simulated trees, writing to a sibling "tsinfer" directory beside that
# run's "sim" directory. Run sim_runs/two_admix_GNN.sh first: the VCFs it writes
# are this run's input.
#
# GNN (not True) is the LAI method here: ground-truth ancestry comes from the
# msprime census nodes recorded at simulation time, and the snakefile rejects
# LAI_method='True' whenever the trees are not the ones this run simulated.
# Inferred trees therefore require an inference-based LAI method.
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

# Extra flags given to this script are forwarded to snakemake (e.g. -n).
SNAKEMAKE_ARGS=("$@")
source "$script_dir/run_inferred.sh"

run_inferred two_admix tsinfer LAI_method=GNN
