#!/bin/bash
# One self-contained SLURM job = one chromosome.
#
#   $1  chromosome (chr1)
#   $2  threads to reserve per run_singer task (1, 2, 4, ...)
#
# The job owns a whole node: 64 cores and, because the partition sets
# DefMemPerNode=UNLIMITED, all 251.6 GB of its memory. `--cores all` picks up the 64 from the
# cgroup mask, and `--set-threads run_singer=N` divides it into 64/N concurrent SINGER tasks.
# That thread count is a MEMORY reservation, not a CPU one -- SINGER is single-threaded, but a
# core is the unit snakemake allocates by, so a task needing more than one core's 3.93 GB share
# has to hold more than one core.
#
# merge_chunks declares no threads, so snakemake runs up to --cores of them at once; each
# loads a whole chromosome ts and AFRICA OOMed 6 chromosomes that way. Capping it at 4 gives
# each merge a quarter of the job memory instead of a sixteenth.
#
# --latency-wait is raised from its 5 s default because merge_chunks gathers every chunk's
# POLEGON output at once: on AMERICA, 4 of 22 chromosomes died there after 9-10 h, each with a
# handful of chunks whose final replicate had not yet appeared. The failure is reported as
# filesystem latency but is really a scheduling race, and it lands late, when the run has the
# most to lose.
set -uo pipefail

chrom=$1
threads=$2
SET=${SET:-OCEANIA}
ROOT=/project/jnovembre/afselim/structure-thru-time/data/human_data

export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
export TCL_LIBRARY="/project/jnovembre/afselim/CoaSTT_env/lib/tcl8.6"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

# Per-chromosome workdir: snakemake keeps run state here and concurrent runs sharing a
# directory race on .snakemake/iocache.
work="${SCRATCH:-/tmp}/${USER}/singer_work/${SET}_${chrom}"
mkdir -p "$work"

echo "=== $(date '+%F %T') ${SET}/${chrom} on $(hostname) ==="
echo "cpus allocated : $(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')"
ncpu=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')
echo "set-threads    : run_singer=${threads}  -> expect $((ncpu / threads)) concurrent SINGER tasks"
echo "workdir        : ${work}"
echo

snakemake \
    --snakefile "${ROOT}/../singer-snakemake/workflow/Snakefile" \
    --configfile "${ROOT}/drivers/singer_config.yaml" \
    --config input-dir="${ROOT}/SingerSnakemake_SampleSets/${SET}" \
             output-dir="${ROOT}/results/${SET}/singer-snakemake_outputs" \
             chromosomes="[\"${chrom}\"]" \
    --directory "$work" \
    --cores all \
    --set-threads run_singer="${threads}" merge_chunks=4 \
    --rerun-incomplete \
    --keep-going \
    --latency-wait 60 \
    --printshellcmds

status=$?
echo
echo "=== $(date '+%F %T') ${chrom} finished with status ${status} ==="
exit $status
