#!/bin/bash
# Run every sim CoaSTT run.
#
#     bash generate_sim_results.sh            # sequentially, on this machine
#     bash generate_sim_results.sh --slurm    # each run as its own SLURM job
#     bash generate_sim_results.sh --slurm -n # print the sbatch plan only
#     bash generate_sim_results.sh -n         # snakemake dry run of each
#
# Each script under CoaSTT_runs/sim_runs/ handles both modes itself: this driver
# only hands every one of them the same flags. Cluster settings come from
# CoaSTT_runs/slurm.yaml (see slurm.yaml.example).
#
# Under --slurm the runs are submitted in two phases, because several
# deliberately share one output_path: the 6 SS p1/p2 pairs, and the admixture
# demographies whose LAI-method variants reuse one simulation. Those runs share
# the upstream stages -- tree simulation, coalescent pdf, genome-wide rates --
# and each run script's comments say the followers reuse whatever runs first.
# Submitted all at once, every member of a group would find the trees missing
# and simulate 500 contigs into the same directory: duplicated compute and
# racing writes to a shared rates.parquet. So:
#
#   phase 1 (leaders)   one run per shared-output group, plus every singleton
#   phase 2 (followers) the rest, held with --dependency=afterok on phase 1
#
# Runs with no shared output are all leaders, so nothing waits needlessly.
set -euo pipefail

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$RUNS_DIR/CoaSTT_runs/sim_runs"
CONFIG_DIR="$RUNS_DIR/../../data/configs/sim_configs"

SLURM=0
PASS_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --slurm) SLURM=1; PASS_ARGS+=("$arg") ;;
        *)       PASS_ARGS+=("$arg") ;;
    esac
done

shopt -s nullglob
scripts=("$RUN_DIR"/*.sh)   # nullglob: expands to nothing when the dir is empty
if [[ ${#scripts[@]} -eq 0 ]]; then
    echo "No run scripts in $RUN_DIR -- nothing to do."
    exit 0
fi

# ------------------------------------------------------------------ local ----
if [[ $SLURM -eq 0 ]]; then
    echo "Running ${#scripts[@]} sim run(s) sequentially."
    for script in "${scripts[@]}"; do
        echo "=== $(basename "$script" .sh) ==="
        bash "$script" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
    done
    echo "All sim runs complete."
    exit 0
fi

# ------------------------------------------------------------------ slurm ----

# The output_path a run writes to, read from the config its snakemake call
# names. Runs sharing a value here contend for the same trees and rates.
output_path_of() {
    local cfg
    cfg=$(grep -oE 'sim_configs/[^ ]+\.yaml' "$1" | head -1)
    [[ -n "$cfg" ]] || { echo "UNKNOWN:$1"; return; }
    grep '^output_path' "$CONFIG_DIR/$(basename "$cfg")" 2>/dev/null | awk '{print $2}'
}

# Prefer the cheapest/most self-sufficient member as a group's leader: SS pairs
# -> p1; admixture -> the True LAI pass where present (it is the 12 h variant
# and needs trees from this pipeline anyway), else the first member.
leader_rank() {
    case "$(basename "$1" .sh)" in
        *_p1)    echo 0 ;;
        *_True)  echo 0 ;;
        *_RFmix) echo 1 ;;
        *)       echo 2 ;;
    esac
}

# Bucket runs by output_path, then split each bucket into leader + followers.
paths=()
for s in "${scripts[@]}"; do paths+=("$(output_path_of "$s")"); done

leaders=(); followers=(); seen=""
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

# Submit one phase, each run held on $dep, and collect the job ids so the next
# phase can depend on them.
#
# A run script reports itself on stderr as `submitted <name>: job <id>` and
# writes nothing to stdout, so the phase's output is captured, echoed back for
# the user, and scanned for those ids. Under -n the ids are placeholders rather
# than numbers, which the [0-9] match skips: a dry run submits nothing, so
# there is nothing for a later phase to depend on.
PHASE_JIDS=""
run_phase() {
    local dep="$1"; shift
    local jids="" script out jid

    for script in "$@"; do
        echo "  $(basename "$script" .sh)"
        out=$(COASTT_DEPENDENCY="$dep" bash "$script" \
                  ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} 2>&1) || {
            echo "$out"
            echo "FAILED: $(basename "$script" .sh)" >&2
            return 1
        }
        echo "$out"
        jid=$(sed -n 's/.*: job \([0-9][0-9]*\).*/\1/p' <<< "$out" | tail -1)
        [[ -n "$jid" ]] && jids="${jids:+$jids:}$jid"
    done

    PHASE_JIDS="$jids"
}

echo "=== Phase 1: ${#leaders[@]} leader run(s) ==="
run_phase "" "${leaders[@]}"
leader_jids="$PHASE_JIDS"

if [[ ${#followers[@]} -gt 0 ]]; then
    echo
    echo "=== Phase 2: ${#followers[@]} follower run(s), held on phase 1 ==="
    run_phase "$leader_jids" "${followers[@]}"
fi

echo
if [[ -n "$leader_jids" ]]; then
    echo "Submitted. Followers start only after every leader job exits 0."
else
    echo "Dry run only -- nothing submitted."
fi
