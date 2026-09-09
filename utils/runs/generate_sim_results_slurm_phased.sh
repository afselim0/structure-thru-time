#!/bin/bash
# Two-phase SLURM submission for the sim runs.
#
# Several runs deliberately share one output_path (Invariant #2 in the refactor
# handoff): the 6 SS p1/p2 pairs, and the admixture demographies whose LAI method
# variants reuse one simulation. Those runs share the upstream stages -- tree
# simulation, coalescent pdf, genome-wide rates -- and each run script's own
# comments say the followers reuse whatever "runs first".
#
# The single-phase driver submits every run as one concurrent array with --nolock,
# so the members of a shared-output group would each find the trees missing and all
# simulate 500 contigs into the same directory at once: duplicated compute and
# racing writes to a shared rates.parquet.
#
# This driver instead submits:
#   phase 1 (leaders)   one run per shared-output group, plus every singleton run
#   phase 2 (followers) the remaining group members, held by --dependency=afterok
#                       on the leader array so trees/rates exist before they start
#
# Runs are grouped by resource profile within each phase, exactly as the original
# driver does, so a light run never inherits a heavy run's allocation.
#
# Usage:  bash generate_sim_results_slurm_phased.sh [-n]
#           -n  print the sbatch commands without submitting
#SBATCH --job-name=coastt_sim
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --output=job_logs/%x_%A_%a.out
#SBATCH --error=job_logs/%x_%A_%a.err
set -euo pipefail

# SLURM runs array tasks from a spooled *copy* of this file, so $0 does not point
# back into the repo. COASTT_RUNS_DIR is exported at submit time and is what an
# array task anchors to; the submitting shell resolves it from $0 as usual.
if [[ -n "${COASTT_RUNS_DIR:-}" ]]; then
    RUNS_DIR_ABS="$COASTT_RUNS_DIR"
else
    RUNS_DIR_ABS="$(cd "$(dirname "$0")" && pwd)"
fi
SELF="$RUNS_DIR_ABS/$(basename "$0")"
cd "$RUNS_DIR_ABS"
RUN_DIR="local_runs/sim_runs"
CONFIG_DIR="../../data/configs/sim_configs"
CONDA_ENV="${COASTT_CONDA_ENV:-/project/jnovembre/afselim/CoaSTT_env}"

# ---------------------------------------------------------------- worker ----
# When SLURM_ARRAY_TASK_ID is set we are an array task: run the script whose
# name was baked into the per-phase manifest passed as $COASTT_MANIFEST.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    manifest="${COASTT_MANIFEST:?COASTT_MANIFEST not set for array task}"
    script=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$manifest")
    [[ -n "$script" ]] || { echo "No script at index $SLURM_ARRAY_TASK_ID in $manifest" >&2; exit 1; }
    echo "Task $SLURM_ARRAY_TASK_ID -> $(basename "$script")"

    # Batch jobs start from a clean shell, so CONDA_EXE is usually unset: source
    # the profile hook explicitly rather than relying on the submitting shell.
    CONDA_SH="${COASTT_CONDA_SH:-/project/jnovembre/afselim/miniforge3/etc/profile.d/conda.sh}"
    if [[ -f "$CONDA_SH" ]]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$CONDA_ENV"
    elif [[ -x "$CONDA_ENV/bin/python" ]]; then
        export PATH="$CONDA_ENV/bin:$PATH"   # prefix env works without the hook
    else
        echo "Cannot activate conda env '$CONDA_ENV' (no $CONDA_SH, no $CONDA_ENV/bin/python)" >&2
        exit 1
    fi

    python -c 'import tskit, msprime, snakemake' \
        || { echo "Env '$CONDA_ENV' is missing pipeline dependencies" >&2; exit 1; }

    # --nolock: array tasks share the utils/ working directory
    bash "$script" --nolock
    exit $?
fi

# ------------------------------------------------------------- submitter ----
DRY=0
[[ "${1:-}" == "-n" ]] && DRY=1

shopt -s nullglob
scripts=("$RUN_DIR"/*.sh)
if [[ ${#scripts[@]} -eq 0 ]]; then
    echo "No run scripts in $RUN_DIR -- nothing to submit."
    exit 0
fi

read_profile() {
    awk -F= '
        /^# CPUS=/{c=$2} /^# MEM_PER_CPU=/{m=$2} /^# TIME=/{t=$2}
        END{print (c?c:12), (m?m:"4G"), (t?t:"12:00:00")}' "$1"
}

# The output_path a run writes to, read from the config its snakemake call names.
# Runs sharing a value here contend for the same simulated trees and rates.
output_path_of() {
    local cfg
    cfg=$(grep -oE '\.\./data/configs/sim_configs/[^ ]+\.yaml' "$1" | head -1)
    [[ -n "$cfg" ]] || { echo "UNKNOWN:$1"; return; }
    grep '^output_path' "$CONFIG_DIR/$(basename "$cfg")" 2>/dev/null | awk '{print $2}'
}

# Prefer the cheapest/most self-sufficient member as a group's leader:
# SS pairs -> p1; admixture -> the True LAI pass where present (it is the 12h
# variant and needs trees from this pipeline anyway), else the first member.
leader_rank() {
    case "$(basename "$1" .sh)" in
        *_p1)    echo 0 ;;
        *_True)  echo 0 ;;
        *_RFmix) echo 1 ;;
        *)       echo 2 ;;
    esac
}

# Manifests are read by each array task when it *starts*, which can be hours after
# submission, so they must outlive the submitting shell. Each submission gets its
# own timestamped directory that is never reused or cleaned up by a later run.
SUBMIT_DIR=".submit/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p job_logs "$SUBMIT_DIR"
leaders=(); followers=()

# Bucket runs by output_path, then split each bucket into leader + followers.
paths=()
for s in "${scripts[@]}"; do paths+=("$(output_path_of "$s")"); done

seen=""
for i in "${!scripts[@]}"; do
    op="${paths[$i]}"
    case " $seen " in *" $op "*) continue ;; esac
    seen="$seen $op"

    members=()
    for j in "${!scripts[@]}"; do
        [[ "${paths[$j]}" == "$op" ]] && members+=("${scripts[$j]}")
    done

    if [[ ${#members[@]} -eq 1 ]]; then
        leaders+=("${members[0]}")
    else
        best="${members[0]}"; best_rank=$(leader_rank "$best")
        for m in "${members[@]:1}"; do
            r=$(leader_rank "$m")
            if [[ $r -lt $best_rank ]]; then best="$m"; best_rank=$r; fi
        done
        leaders+=("$best")
        for m in "${members[@]}"; do
            [[ "$m" != "$best" ]] && followers+=("$m")
        done
    fi
done

# Submit one array per resource profile within a phase. Writes a manifest so each
# array task maps its index to a script without re-deriving the grouping.
submit_phase() {
    local phase="$1"; shift
    local dep="$1"; shift
    local -a phase_scripts=("$@")
    local jids=""

    [[ ${#phase_scripts[@]} -eq 0 ]] && { echo ""; return; }

    local profs=()
    for s in "${phase_scripts[@]}"; do profs+=("$(read_profile "$s")"); done

    local uniq=()
    while IFS= read -r line; do uniq+=("$line"); done \
        < <(printf '%s\n' "${profs[@]}" | awk '!seen[$0]++')

    for prof in "${uniq[@]}"; do
        read -r cpus mem time <<< "$prof"
        # Absolute path: the array task resolves this from its own working directory.
        local manifest="$PWD/$SUBMIT_DIR/${phase}_$(echo "$prof" | tr ' ' '_').txt"
        : > "$manifest"
        local n=0
        for i in "${!phase_scripts[@]}"; do
            if [[ "${profs[$i]}" == "$prof" ]]; then
                echo "${phase_scripts[$i]}" >> "$manifest"
                n=$((n + 1))
            fi
        done

        local depflag=()
        [[ -n "$dep" ]] && depflag=(--dependency=afterok:"$dep")

        if [[ $DRY -eq 1 ]]; then
            echo "[dry] sbatch --array=1-$n --cpus-per-task=$cpus --mem-per-cpu=$mem --time=$time ${depflag[*]:-} $SELF" >&2
            { echo "      manifest $manifest:"; sed "s/^/        /" "$manifest"; } >&2
        else
            local jid
            jid=$(sbatch --parsable --array="1-$n" \
                         --cpus-per-task="$cpus" --mem-per-cpu="$mem" --time="$time" \
                         --export=ALL,COASTT_MANIFEST="$manifest",COASTT_RUNS_DIR="$RUNS_DIR_ABS" \
                         "${depflag[@]}" "$SELF")
            echo "Submitted $phase array: job $jid ($n tasks; ${cpus} cpus, ${mem}/cpu, ${time})" >&2
            jids="${jids:+$jids:}$jid"
        fi
    done
    echo "$jids"
}

echo "=== Phase 1: ${#leaders[@]} leader run(s) ==="
for s in "${leaders[@]}"; do echo "  $(basename "$s" .sh)"; done
leader_jids=$(submit_phase leaders "" "${leaders[@]}")

echo
echo "=== Phase 2: ${#followers[@]} follower run(s), held on phase 1 ==="
for s in "${followers[@]}"; do echo "  $(basename "$s" .sh)"; done
submit_phase followers "$leader_jids" "${followers[@]}" > /dev/null

echo
if [[ $DRY -eq 1 ]]; then
    echo "Dry run only -- nothing submitted."
else
    echo "Submitted. Followers start only after every leader array task exits 0."
fi
