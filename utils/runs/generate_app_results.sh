#!/bin/bash
# Run every app CoaSTT run.
#
#     bash generate_app_results.sh            # sequentially, on this machine
#     bash generate_app_results.sh --slurm    # each run as its own SLURM job
#     bash generate_app_results.sh --slurm -n # print the sbatch plan only
#     bash generate_app_results.sh -n         # snakemake dry run of each
#
# Each script under CoaSTT_runs/app_runs/ handles both modes itself: this driver
# only hands every one of them the same flags. Cluster settings come from
# CoaSTT_runs/slurm.yaml (see slurm.yaml.example).
#
# Unlike the sim runs, these need no phasing. Every app run writes to its own
# output_path, and the two that do have internal ordering -- chimps.sh, whose
# three configs share one pdf, and arg_rates_comparison.sh, whose method stages
# need the sim stage's trees -- chain their own jobs when given --slurm.
#
# The SINGER runs do depend on the matching sim run's simulated trees, but on
# the sim runs rather than on each other: run generate_sim_results.sh first.
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$RUNS_DIR/CoaSTT_runs/app_runs"

shopt -s nullglob
scripts=("$RUN_DIR"/*.sh)   # nullglob: expands to nothing when the dir is empty
if [[ ${#scripts[@]} -eq 0 ]]; then
    echo "No run scripts in $RUN_DIR -- nothing to do."
    exit 0
fi

# Skip placeholders: a script with no runnable content would otherwise be
# reported as a run that succeeded instantly.
runnable=()
for s in "${scripts[@]}"; do
    grep -qE '^[^#[:space:]]' "$s" && runnable+=("$s")
done
if [[ ${#runnable[@]} -eq 0 ]]; then
    echo "No runnable scripts in $RUN_DIR -- nothing to do."
    exit 0
fi

echo "Running ${#runnable[@]} app run(s)."
for script in "${runnable[@]}"; do
    echo "=== $(basename "$script" .sh) ==="
    bash "$script" "$@"
done
echo "All app runs submitted/complete."
