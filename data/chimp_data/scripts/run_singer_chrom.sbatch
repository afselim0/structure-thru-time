#!/bin/bash
# One SLURM job per chromosome, running the WHOLE workflow inside it with local parallelism.
#
# Submitted by scripts/run_singer.sh -- not usually run by hand.
#
# WHY THIS SHAPE. singer-snakemake is designed to run as `snakemake --cores=N` on one machine
# (its README quickstart is `--cores=20`); SLURM support is a documented extra, not the model
# it was built around. That matters because `run_polegon` is parameterised per (chunk, draw):
# at 897 chunks x 20 draws it is 17,940 invocations of a binary that takes seconds. As local
# work items draining across cores that is a work queue. As individual SLURM submissions it is
# pathological -- scheduling and teardown cost more than the compute, and throughput becomes
# submission-rate-bound rather than CPU-bound (measured on the human side: ~3,700 jobs/h,
# ~32 h of wall clock for ~4 h of compute).
#
# So: one job per chromosome, 48 cores, and snakemake schedules its own rules inside it with
# no executor at all. The 897 chunks split across 23 chromosomes, the largest being chr1 at
# 75 chunks, so a 48-core node keeps every core busy without the chunk count wildly exceeding
# what one job can drain.
#
# Resources: a WHOLE jnovembre node per chromosome. 64 cores, 252 GB -- about 3.93 GB per
# core, which is the budget each single-threaded SINGER process gets when all 64 run at once.
#
# SINGER is single-threaded and no rule in the workflow declares `threads:`, so one core per
# chunk is the natural unit and --cores 64 keeps every core busy. Reserving the whole node
# (--exclusive) rather than 64 shared cores matters because the memory budget above assumes
# the full 252 GB is ours.
#
# Measured at 3 Mb chunks, n=2,580: memory p50 0.67 GB, p99 3.25 GB, MAX 4.00 GB -- the worst
# case exceeded a core's 3.93 GB share, which is why chunks are now 2 Mb (see the note in
# drivers/singer_config.yaml). Runtime p50 0.94 h, p90 9.95 h, max 19.81 h; a node's job ends
# when its SLOWEST chunk does, so 35 h covers the tail with room after the 2 Mb reduction.
#SBATCH --job-name=chimp_singer
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=35:00:00
#SBATCH --output=logs/chimp_singer-%x-%j.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

CHROM="$1"
ROOT="$2"
SINGER="${ROOT}/../singer-snakemake"
WORK="$3"

mkdir -p "$WORK"
echo "START ${CHROM} $(date -Is) on $(hostname) with ${SLURM_CPUS_PER_TASK} cores"

# No --profile and no --executor: snakemake runs every rule as a local subprocess, scheduling
# them across --cores. This is the point of the whole arrangement.
snakemake \
    --snakefile "${SINGER}/workflow/Snakefile" \
    --configfile "${ROOT}/drivers/singer_config.yaml" \
    --config input-dir="${ROOT}/SingerSnakemake_Input" \
             output-dir="${ROOT}/results/chimp/singer-snakemake_outputs" \
             chromosomes="[\"${CHROM}\"]" \
    --directory "$WORK" \
    --cores "$SLURM_CPUS_PER_TASK" \
    --rerun-incomplete \
    --nolock

echo "END ${CHROM} $(date -Is)"
