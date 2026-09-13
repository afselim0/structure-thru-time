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
# Resources are sized from measurement, not assumption:
#   * run_singer   p50 1.3 h, p90 11.5 h, p99 18.3 h per chunk (n=7,255 completed)
#   * memory       2.4-4.5 GB per concurrent chunk; 48 x 4.5 GB would exceed a node, so
#                  --cores is what bounds concurrency and 180 GB is the node's ceiling anyway
#   * walltime     chr1's 75 chunks over 48 cores is ~2 sequential rounds; at p90 that is
#                  ~23 h, and the caslake QOS ceiling is 36 h
#SBATCH --job-name=chimp_singer
#SBATCH --account=pi-jnovembre
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
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
