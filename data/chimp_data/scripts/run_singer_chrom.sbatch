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
# Resources: a WHOLE node per chromosome, reserved exclusively.
#
# SINGER is single-threaded and no rule in the workflow declares `threads:`, so one core per
# chunk is the natural unit and `--cores all` keeps every core on the node busy. Reserving the
# node exclusively matters because the memory budget is the node's, not a share of it:
#   caslake    48 cores / 180 GB -> 3.75 GB per concurrent chunk
#   jnovembre  64 cores / 252 GB -> 3.93 GB per concurrent chunk
#
# Measured at 3 Mb chunks (n=2,580 completed): memory p50 0.67 GB, p99 3.25 GB, MAX 4.00 GB --
# the worst case exceeded a core's share, which is why chunks are now 2 Mb (see the note in
# drivers/singer_config.yaml). Runtime at 3 Mb was p50 0.94 h, p90 9.95 h, max 19.81 h, and a
# node's job ends when its SLOWEST chunk does, so the tail is what the walltime must cover.
#
# 35 h sits just under the caslake QOS ceiling of 1-12:00:00 (36 h); a request AT the ceiling
# is rejected rather than truncated. jnovembre's 48 h QOS leaves more room.
#SBATCH --job-name=chimp_singer
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --nodes=1
#SBATCH --exclusive
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
echo "START ${CHROM} $(date -Is) on $(hostname) with $(nproc) cores"

# No --profile and no --executor: snakemake runs every rule as a local subprocess, scheduling
# them across --cores. This is the point of the whole arrangement.
snakemake \
    --snakefile "${SINGER}/workflow/Snakefile" \
    --configfile "${ROOT}/drivers/singer_config.yaml" \
    --config input-dir="${ROOT}/SingerSnakemake_Input" \
             output-dir="${ROOT}/results/chimp/singer-snakemake_outputs" \
             chromosomes="[\"${CHROM}\"]" \
    --directory "$WORK" \
    --cores all \
    --rerun-incomplete \
    --nolock

echo "END ${CHROM} $(date -Is)"
