#!/usr/bin/env bash
# Resubmit per-chromosome SINGER jobs that died of a wall-clock TIMEOUT.
#
#   bash scripts/resubmit_timeouts.sh EUROPE jnovembre 16 2 48:00:00
#   bash scripts/resubmit_timeouts.sh EUROPE jnovembre 16 2 48:00:00 --watch
#
# Resuming is safe and cheap: the workflow's per-chunk `.singer.done` sentinels
# survive the kill, so a resubmitted job redoes only the chunks that had not
# finished, not the inference behind the ones that had. A timeout therefore costs
# queue time, never compute.
#
# Only TIMEOUT is resubmitted. COMPLETED needs nothing, and CANCELLED means a
# person intervened -- relaunching that would fight the operator. FAILED and
# NODE_FAIL are reported but NOT resubmitted, because a job that fails fast will
# fail fast again and an automatic loop would spin.
#
# Without --watch it makes one pass and exits, which is the form to use from a
# terminal. With --watch it polls until no job of this set is left queued or
# running, which is the form to leave in the background.
set -uo pipefail

SET=${1:?usage: resubmit_timeouts.sh SET PARTITION CPUS THREADS WALLTIME [--watch]}
PARTITION=${2:?}
CPUS=${3:?}
THREADS=${4:?}
WALLTIME=${5:?}
WATCH=${6:-}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT=pi-jnovembre
# Job names are what ties a SLURM job back to a chromosome, so they must match
# the ones used at first submission.
prefix=$(echo "${SET:0:2}" | tr '[:upper:]' '[:lower:]')

submit_one() {
    local chrom=$1
    sbatch --parsable \
        --job-name="${prefix}_${chrom}" \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS}" \
        --time="${WALLTIME}" \
        --output="${ROOT}/logs/${SET}_${chrom}.out" \
        --error="${ROOT}/logs/${SET}_${chrom}.err" \
        --export=ALL,SET="${SET}" \
        "${ROOT}/scripts/run_singer_chrom.sh" "${chrom}" "${THREADS}"
}

one_pass() {
    local resubmitted=0
    for c in $(seq 1 22); do
        local chrom="chr${c}"
        local name="${prefix}_${chrom}"

        # Already queued or running: leave it alone. This is what stops a watch
        # loop from submitting the same chromosome twice.
        if squeue -u "$USER" -h -n "$name" -o "%i" | grep -q .; then
            continue
        fi

        # Finished chromosomes have their full set of tree sequences. Checking the
        # OUTPUT rather than the exit code means an already-good chromosome is
        # never relaunched, whatever sacct says about how it ended.
        local trees="${ROOT}/results/${SET}/singer-snakemake_outputs/${chrom}/trees"
        local n_trees=0
        [[ -d "$trees" ]] && n_trees=$(ls "$trees"/*.tsz 2>/dev/null | wc -l)
        if [[ "$n_trees" -ge 20 ]]; then
            continue
        fi

        # Most recent terminal state for this job name.
        local state
        state=$(sacct -u "$USER" -n -X --name="$name" \
                      -o State --starttime=now-3days 2>/dev/null \
                | awk 'NF{s=$1} END{print s}')

        case "$state" in
            TIMEOUT)
                local jid
                jid=$(submit_one "$chrom")
                echo "$(date '+%F %T') ${chrom}: TIMEOUT at ${n_trees}/20 trees -> resubmitted as ${jid} (${WALLTIME})"
                resubmitted=$((resubmitted + 1))
                ;;
            FAILED|NODE_FAIL|OUT_OF_MEMORY)
                echo "$(date '+%F %T') ${chrom}: ${state} at ${n_trees}/20 trees -- NOT resubmitted, needs a look"
                ;;
            CANCELLED*)
                echo "$(date '+%F %T') ${chrom}: CANCELLED -- left alone (assumed deliberate)"
                ;;
            "")
                ;;
            *)
                echo "$(date '+%F %T') ${chrom}: ${state} at ${n_trees}/20 trees -- no action"
                ;;
        esac
    done
    return $resubmitted
}

if [[ "$WATCH" != "--watch" ]]; then
    one_pass
    exit 0
fi

echo "$(date '+%F %T') watching ${SET} on ${PARTITION}; resubmitting timeouts at ${WALLTIME}"
while true; do
    one_pass
    # Stop once nothing of this set is queued or running any more: either every
    # chromosome finished, or what is left needs a human.
    if ! squeue -u "$USER" -h -o "%j" | grep -q "^${prefix}_chr"; then
        echo "$(date '+%F %T') no ${SET} jobs left queued or running; watch ends"
        break
    fi
    sleep 600
done
