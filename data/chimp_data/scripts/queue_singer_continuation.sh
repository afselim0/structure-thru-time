#!/usr/bin/env bash
# Queue a CONTINUATION job for each running singer chromosome job.
#
#   bash scripts/queue_singer_continuation.sh -n        # dry run: show what would be submitted
#   bash scripts/queue_singer_continuation.sh           # submit
#   bash scripts/queue_singer_continuation.sh --time 36:00:00
#
# WHY. The first wave of jobs was sized at 20 h, which measurement showed is short for the
# large chromosomes: per-chunk runtime at 2 Mb came out p50 4.6 h, max 5.9 h, so a 16-core job
# clears ~16 chunks per ~5 h round. chr1 (112 chunks) needs 7 rounds, ~32 h. Chromosomes above
# ~5 rounds will hit the wall before finishing.
#
# Rather than cancelling and resubmitting -- which throws away the hours already invested --
# each running job gets a dependent successor that starts the moment it ends. The successor
# re-runs the identical snakemake command, which RESUMES rather than restarts: completed
# chunks are recorded by `.singer.done` sentinels under results/, and --rerun-incomplete
# discards only the partial work of chunks that were in flight when the wall hit.
#
# afterany, NOT afterok. A job killed at its walltime exits non-zero. Under afterok the
# successor would sit in DependencyNeverSatisfied forever -- precisely the case this exists to
# handle. afterany fires on any terminal state, including TIMEOUT, so a chromosome that walls
# is picked straight back up, and one that finishes cleanly leaves its successor a no-op that
# exits in seconds once snakemake sees nothing left to do.
#
# Cost of a no-op successor: one short job per already-finished chromosome. That is cheaper
# than predicting which chromosomes will wall and being wrong.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRYRUN=0
PARTITION="caslake"
TIME="20:00:00"
WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRATCH:-/tmp}/${USER}/singer_work}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --partition) PARTITION=$2; shift 2 ;;
        --time)      TIME=$2; shift 2 ;;
        -n|--dry-run) DRYRUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Only chromosomes with a job CURRENTLY in the queue get a successor. A chromosome whose job
# has already left the queue has nothing to depend on; submitting an independent job for it
# here would race against nothing but would also start immediately, which is not what this
# script is for -- resubmit those with run_singer.sh instead.
#
# A PENDING predecessor is a perfectly good thing to chain onto: SLURM holds the successor
# until the predecessor reaches a terminal state, whenever that is. So this works the same
# whether the current layer is running or still queued.
#
# Matches jobs from ANY layer -- singer_chr1, singer_chr1_cont, singer_chr1_cont2, ... -- so
# the script can be run repeatedly to extend the chain. Successors are tagged with the layer
# number so `squeue` shows which generation a job belongs to, and so a job name never
# collides with its own predecessor.
mapfile -t ROWS < <(squeue -u "$USER" -h -o "%i %j" | grep -E ' singer_chr[0-9A-Z]+(_cont[0-9]*)?$' | sort -k2 -V)
if [[ ${#ROWS[@]} -eq 0 ]]; then
    echo "no singer_chr* jobs in the queue -- nothing to chain onto" >&2
    exit 1
fi

mkdir -p "${ROOT}/logs"
submitted=0
for row in "${ROWS[@]}"; do
    jid=${row%% *}
    name=${row##* }
    # Strip the layer suffix to recover the chromosome: singer_chr1_cont2 -> chr1.
    c=${name#singer_}
    c=${c%%_cont*}
    work="${WORKDIR_ROOT}/chimp/${c}"

    # Next layer number: singer_chr1 -> _cont, _cont -> _cont2, _cont2 -> _cont3.
    case "$name" in
        *_cont)  suffix="_cont2" ;;
        *_cont[0-9]*) suffix="_cont$(( ${name##*_cont} + 1 ))" ;;
        *)       suffix="_cont" ;;
    esac

    if [[ $DRYRUN -eq 1 ]]; then
        echo "would submit ${c}${suffix}: depends afterany:${jid} (${name}), ${TIME} on ${PARTITION}"
        continue
    fi

    new=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="singer_${c}${suffix}" \
        --dependency="afterany:${jid}" \
        --time="$TIME" \
        "${ROOT}/scripts/run_singer_chrom.sbatch" "$c" "$ROOT" "$work" \
        | tail -1)
    echo "queued ${c} as ${new}  (after ${jid})"
    submitted=$((submitted + 1))
done

if [[ $DRYRUN -eq 1 ]]; then
    echo
    echo "Dry run -- nothing submitted."
    exit 0
fi

echo
echo "Queued ${submitted} continuation job(s) on ${PARTITION} at ${TIME}."
echo "They stay PENDING with reason (Dependency) until their predecessor ends."
echo "  squeue -u \$USER -o '%.10i %.20j %.2t %.10M %R' | grep singer_"
