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
# Resources: 16 CPUs on ONE node, sharing the node with other jobs.
#
# SINGER is single-threaded and no rule in the workflow declares `threads:`, so one core per
# chunk is the natural unit: snakemake runs 16 chunks in parallel per chromosome, 23
# chromosomes at once.
#
# --cores takes $SLURM_CPUS_PER_TASK, NOT `all`. On a shared node `all` resolves to the
# node's physical core count (48 on caslake), not the 16 CPUs this job was allocated, so
# snakemake would start 48 chunks inside a 16-CPU cgroup -- 3x oversubscribed, every chunk
# slower, and peak memory sized for 48 concurrent chunks rather than 16.
#
# --nodes=1 with --ntasks=1 keeps all 16 CPUs on a single node. Snakemake schedules its rules
# as local subprocesses, so a split allocation across two nodes would leave half the cores
# unreachable.
#
# Not --exclusive: 16 of a caslake node's 48 cores, so three chromosomes can share a node and
# the jobs backfill into partial nodes instead of waiting for whole ones. Memory comes from
# SLURM's per-CPU default for the partition rather than being requested explicitly.
#
# Walltime: 20 h. Measured at 3 Mb chunks (n=2,580) the per-chunk runtime was p50 0.94 h,
# p90 9.95 h, max 19.81 h; chunks are now 2 Mb, and cost is roughly linear in chunk length,
# so that tail should compress. A job ends when its SLOWEST chunk does.
#
# A walltime kill loses the whole chunk rather than retrying, so if chromosomes start hitting
# this, raise it rather than letting them die -- the caslake QOS allows up to 1-12:00:00
# (36 h), though a request AT the ceiling is rejected rather than truncated.
#SBATCH --job-name=chimp_singer
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
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
