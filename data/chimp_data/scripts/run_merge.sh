#!/usr/bin/env bash
# Re-run merge_chunks for chromosomes whose merge was OOM-killed, with parallelism capped.
#
#   bash scripts/run_merge.sh -n                 # dry run
#   bash scripts/run_merge.sh                    # submit
#   bash scripts/run_merge.sh --chrom chr6       # one chromosome
#
# Chromosomes whose SINGER job is STILL RUNNING get a dependent merge job (afterany) rather
# than an immediate one: two snakemakes on the same chromosome would race on the same output
# tree even with separate workdirs, and the merge cannot start until the last chunk exists
# anyway.
#
# Everything else is submitted immediately. Completed chunks persist as .singer.done sentinels,
# so this recomputes no SINGER work -- it writes only the draws the killed job never reached.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRYRUN=0
CHROM=""
PARTITION="caslake"
WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRATCH:-/tmp}/${USER}/singer_work}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --chrom)     CHROM=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        -n|--dry-run) DRYRUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

OUT="${ROOT}/results/chimp/singer-snakemake_outputs"
if [[ -n "$CHROM" ]]; then
    CHROMS=("$CHROM")
else
    mapfile -t CHROMS < <(cut -f1 "${ROOT}/resources/panTro6/chrom.sizes")
fi

mkdir -p "${ROOT}/logs"
submitted=0
# `set -e` off for the submission loop. Two reasons: the per-chromosome probes (find on a
# missing trees/, squeue matching nothing) legitimately return non-zero, and one chromosome
# failing to submit should not abort the remaining 22 -- each is independent work.
set +e
for c in "${CHROMS[@]}"; do
    [[ -d "${OUT}/${c}" ]] || continue
    draws=$(find "${OUT}/${c}/trees" -maxdepth 1 -name "*.tsz" 2>/dev/null | wc -l)
    # 20 draws == singer-mcmc-samples; a chromosome at 20 is finished and needs nothing.
    if [[ $draws -ge 20 ]]; then
        echo "skip ${c}: already ${draws}/20 draws"
        continue
    fi

    # Chain onto a still-running SINGER job for this chromosome, if there is one.
    # Two `set -euo pipefail` traps here, both hit in testing:
    #   * grep exits 1 when the chromosome has no running job -- the normal case;
    #   * anything that closes the pipe early (`head -1`, or `exit` inside awk) makes squeue
    #     take SIGPIPE, which pipefail reports as a failure.
    # So: no early exit. awk reads all of squeue's output and keeps only the first match.
    dep=$(squeue -u "$USER" -h -o "%i %j" \
          | awk -v c="$c" '!f && $2 ~ "^singer_" c "(_cont[0-9]*)?$" {d=$1; f=1} END {print d}')

    work="${WORKDIR_ROOT}/chimp/${c}"
    if [[ $DRYRUN -eq 1 ]]; then
        echo "would submit ${c}: ${draws}/20 draws${dep:+, depends afterany:${dep}}"
        continue
    fi

    new=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="merge_${c}" \
        ${dep:+--dependency=afterany:${dep}} \
        "${ROOT}/scripts/run_merge_chrom.sbatch" "$c" "$ROOT" "$work" \
        | tail -1)
    echo "submitted merge ${c} as ${new}  (${draws}/20 draws${dep:+, after ${dep}})"
    submitted=$((submitted + 1))
done

set -e
[[ $DRYRUN -eq 1 ]] && { echo; echo "Dry run -- nothing submitted."; exit 0; }
echo
echo "Submitted ${submitted} merge job(s) to ${PARTITION}."
