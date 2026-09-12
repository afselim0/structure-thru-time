#!/bin/bash
#SBATCH --job-name=chimp_singer
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --cpus-per-task=16
#SBATCH --mem=56G
#SBATCH --time=36:00:00
#SBATCH --array=1-23
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

SUB="$SLURM_SUBMIT_DIR"
CHROM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUB/resources/panTro6/chrom.sizes" | cut -f1)

# Each chromosome gets its own snakemake workdir: concurrent snakemakes in one directory
# race on .snakemake/iocache even with --nolock. The workdir is a symlink farm back to the
# clone so the workflow, scripts and bundled binaries are shared rather than copied.
WORK="$SUB/runs/work/$CHROM"
mkdir -p "$WORK"
for item in workflow resources config; do
  [ -e "$WORK/$item" ] || ln -s "$SUB/../singer-snakemake/$item" "$WORK/$item"
done

# Per-chromosome config: identical to production except it targets one chromosome.
CFG="$WORK/config.yaml"
sed -e "s|^chromosomes:.*|chromosomes: [\"$CHROM\"]|" \
    -e "s|^input-dir:.*|input-dir: \"$SUB/SingerSnakemake_Input\"|" \
    -e "s|^output-dir:.*|output-dir: \"$SUB/results/singer-snakemake_outputs\"|" \
    "$SUB/singer_config.yaml" > "$CFG"

cd "$WORK"
echo "START $CHROM $(date -Is)"
snakemake --configfile "$CFG" --cores "$SLURM_CPUS_PER_TASK" --rerun-incomplete --nolock
echo "END $CHROM $(date -Is)"
