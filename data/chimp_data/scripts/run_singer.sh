#!/usr/bin/env bash
# Run singer-snakemake on the chimpanzee panel: ONE SLURM JOB PER CHROMOSOME.
#
#   bash scripts/run_singer.sh                          # all 23 autosomes
#   bash scripts/run_singer.sh --chrom chr21            # one chromosome
#   bash scripts/run_singer.sh --partition jnovembre    # dedicated partition instead of caslake
#   bash scripts/run_singer.sh -n                       # dry run: show what would be submitted
#
# WHY ONE JOB PER CHROMOSOME, rather than letting snakemake submit each rule as its own job.
#
# singer-snakemake is written to run as `snakemake --cores=N` on a single machine -- that is
# its README quickstart, and SLURM is a documented extra rather than the model it was designed
# around. The consequence shows up in `run_polegon`, which is parameterised per (chunk, draw):
# 897 chunks x 20 draws = 17,940 invocations of a binary that runs in seconds. Locally that is
# a work queue draining across cores. As individual SLURM submissions it is pathological --
# scheduler overhead exceeds the compute, and throughput is bound by the submission rate
# rather than by CPUs (measured: ~3,700 jobs/h, ~32 h wall clock for ~4 h of work).
#
# So each chromosome gets one 16-CPU job and snakemake schedules its own rules inside it as
# local subprocesses, with no executor. 23 jobs instead of ~20,000. SINGER is single-threaded,
# so one core per chunk means 16 chunks in parallel per chromosome.
#
# Produces:
#   results/chimp/singer-snakemake_outputs/<chrom>/   the workflow's own output tree
#   results/chimp/ts_chunks/draw_<rep>/               built afterwards by scripts/extract_chunks.sh
#
# Each chromosome gets its own snakemake workdir under scratch: concurrent snakemakes sharing
# one directory race on .snakemake/iocache, and --nolock is not sufficient. That state is
# disposable -- which chunks are finished is recorded by the `.singer.done` sentinels under
# results/, so deleting a workdir loses no completed work and a restart resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHROM=""
DRYRUN=0
PARTITION="caslake"
# $SCRATCH on midway3 is the SHARED root (/scratch/midway3), not a per-user directory, so the
# user component has to be appended explicitly.
WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRATCH:-/tmp}/${USER}/singer_work}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --chrom)     CHROM=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        -n|--dry-run) DRYRUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

input="${ROOT}/SingerSnakemake_Input"
[[ -d "$input" ]] || { echo "no input directory at ${input} -- run drivers 1-7 first" >&2; exit 1; }

if [[ -n "$CHROM" ]]; then
    CHROMS=("$CHROM")
else
    mapfile -t CHROMS < <(cut -f1 "${ROOT}/resources/panTro6/chrom.sizes")
fi

mkdir -p "${ROOT}/logs"
submitted=0
for c in "${CHROMS[@]}"; do
    [[ -f "${input}/${c}.vcf.gz" ]] || { echo "skip ${c}: no ${c}.vcf.gz" >&2; continue; }
    work="${WORKDIR_ROOT}/chimp/${c}"

    if [[ $DRYRUN -eq 1 ]]; then
        echo "would submit ${c}  (workdir ${work}, partition ${PARTITION})"
        continue
    fi

    jid=$(sbatch --parsable \
        --partition="$PARTITION" \
        --job-name="singer_${c}" \
        "${ROOT}/scripts/run_singer_chrom.sbatch" "$c" "$ROOT" "$work" \
        | tail -1)
    echo "submitted ${c} as ${jid}"
    submitted=$((submitted + 1))
done

if [[ $DRYRUN -eq 1 ]]; then
    echo
    echo "Dry run -- nothing submitted."
    exit 0
fi

echo
echo "Submitted ${submitted} chromosome job(s) to ${PARTITION}. Monitor with:"
echo "  squeue -u \$USER -n singer_<chrom>"
echo "  tail -f ${ROOT}/logs/chimp_singer-*.out"
echo
echo "When every chromosome finishes, build the CoaSTT input directory:"
echo "  bash ${ROOT}/scripts/extract_chunks.sh"
