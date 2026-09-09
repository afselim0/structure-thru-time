#!/bin/bash
# Submit every app CoaSTT run as one SLURM job array (one task per run script).
#
# Usage:  sbatch generate_app_results_slurm.sh
#
# Invoked without SLURM_ARRAY_TASK_ID it acts as the submitter: it counts the run
# scripts and resubmits itself as an array. Each array task then executes the run
# script for its index, using that script's own CPUS/MEM_PER_CPU/TIME hints.
#
# NOTE: the cluster settings below are placeholders carried over from the previous
# slurm scripts and MUST be reviewed before first use on a cluster.
#SBATCH --job-name=coastt_app
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --output=job_logs/%x_%A_%a.out
#SBATCH --error=job_logs/%x_%A_%a.err
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"
RUN_DIR="local_runs/app_runs"
CONDA_ENV="${COASTT_CONDA_ENV:-/project/jnovembre/afselim/CoaSTT_env}"

shopt -s nullglob
# nullglob: expands to nothing when the dir is empty. *_slurm.sh are submitters
# of their own (e.g. arg_rates_comparison_slurm.sh), not array run scripts.
scripts=()
for s in "$RUN_DIR"/*.sh; do
    [[ "$s" == *_slurm.sh ]] || scripts+=("$s")
done

if [[ ${#scripts[@]} -eq 0 ]]; then
    echo "No run scripts in $RUN_DIR — nothing to submit."
    exit 0
fi

# Read the CPUS/MEM_PER_CPU/TIME hints out of a run script.
read_profile() {
    awk -F= '
        /^# CPUS=/{c=$2} /^# MEM_PER_CPU=/{m=$2} /^# TIME=/{t=$2}
        END{print (c?c:12), (m?m:"4G"), (t?t:"12:00:00")}' "$1"
}

# --- Submitter -----------------------------------------------------------
# Runs are grouped by resource profile and each group is submitted as its own
# array, so a light run never inherits a heavy run's allocation (or vice versa).
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    mkdir -p job_logs

    profiles=()
    for s in "${scripts[@]}"; do
        profiles+=("$(read_profile "$s")")
    done

    # Unique profiles, preserving order of first appearance
    uniq_profiles=()
    while IFS= read -r line; do
        uniq_profiles+=("$line")
    done < <(printf '%s\n' "${profiles[@]}" | awk '!seen[$0]++')

    for prof in "${uniq_profiles[@]}"; do
        read -r cpus mem time <<< "$prof"
        # 1-based indices of the runs sharing this profile
        idx=()
        for i in "${!profiles[@]}"; do
            [[ "${profiles[$i]}" == "$prof" ]] && idx+=("$((i + 1))")
        done
        list=$(IFS=,; echo "${idx[*]}")
        jid=$(sbatch --parsable --array="$list" \
                     --cpus-per-task="$cpus" --mem-per-cpu="$mem" --time="$time" \
                     "$SELF")
        echo "Submitted app array: job $jid (${#idx[@]} tasks; ${cpus} cpus, ${mem}/cpu, ${time})"
    done
    exit 0
fi

# --- Worker --------------------------------------------------------------
script="${scripts[$SLURM_ARRAY_TASK_ID - 1]}"
echo "Task $SLURM_ARRAY_TASK_ID -> $(basename "$script")"

if [[ -n "${CONDA_EXE:-}" ]]; then
    source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    # snakemake's highspy ships a libstdc++ that shadows the env's; numpy then
    # fails with a GLIBCXX error unless the env's lib comes first.
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# --nolock: array tasks share the utils/ working directory
bash "$script" --nolock
