#!/bin/bash
# Re-run the tail of the workflow for one chromosome with merge_chunks parallelism capped.
#
# Submitted by scripts/run_merge.sh -- not usually run by hand.
#
# WHY THIS EXISTS. merge_chunks declares no `threads:` and no `resources:` in the upstream
# Snakefile, so snakemake treats each (chunk, draw) merge as a 1-core job and will run as many
# concurrently as --cores allows. Each merge loads a whole chromosome's tree sequence, so on
# the large chromosomes 16 concurrent merges exceeded the cgroup: measured 60.6 GB peak against
# a 59.5 GB limit (16 CPUs x 3810 MB default), killing chr1/chr5/chr6/chr7/chr8/chr10 partway
# through their draws. The small chromosomes merged fine at the same settings, because the
# footprint scales with chromosome size, not chunk count.
#
# --set-threads merge_chunks=4 makes each merge claim 4 of the 16 cores, so at most 4 run at
# once. That is a 4x cut in peak memory for the merge phase and leaves SINGER (which declares
# no threads either, but is single-threaded and small) unaffected for any chunks still to run.
#
# --mem is ALSO raised rather than relying on the thread cap alone: the cap bounds concurrency,
# but a single chromosome-scale merge is itself large, and the default per-CPU allocation was
# never sized for it. Belt and braces -- an OOM here costs a whole job.
#
# Walltime 20 h, NOT the 6 h this script first used. The merge itself is short, but snakemake
# builds the whole DAG: a chromosome still missing a chunk runs SINGER for it FIRST, and a
# single 2 Mb chunk measured p50 4.6 h / max 5.9 h. chr4 was killed at exactly 6 h having spent
# the entire allocation on one chunk with no time left to merge, losing that chunk's work too.
# So the walltime has to cover (slowest outstanding chunk + merge), not the merge alone.
#SBATCH --job-name=chimp_merge
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=20:00:00
#SBATCH --output=logs/chimp_merge-%x-%j.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

CHROM="$1"
ROOT="$2"
SINGER="${ROOT}/../singer-snakemake"
WORK="$3"

mkdir -p "$WORK"
echo "START merge ${CHROM} $(date -Is) on $(hostname) with ${SLURM_CPUS_PER_TASK} cores, mem ${SLURM_MEM_PER_NODE:-unset}"

snakemake \
    --snakefile "${SINGER}/workflow/Snakefile" \
    --configfile "${ROOT}/drivers/singer_config.yaml" \
    --config input-dir="${ROOT}/SingerSnakemake_Input" \
             output-dir="${ROOT}/results/chimp/singer-snakemake_outputs" \
             chromosomes="[\"${CHROM}\"]" \
    --directory "$WORK" \
    --cores "$SLURM_CPUS_PER_TASK" \
    --set-threads merge_chunks=4 \
    --rerun-incomplete \
    --nolock

echo "END merge ${CHROM} $(date -Is)"
