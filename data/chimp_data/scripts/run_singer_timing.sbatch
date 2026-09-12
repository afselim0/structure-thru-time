#!/bin/bash
#SBATCH --job-name=chimp_timing21
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR/../singer-snakemake"
echo "START $(date -Is)"
snakemake --configfile ../chimp_data/singer_config_timing.yaml \
          --cores "$SLURM_CPUS_PER_TASK" --rerun-incomplete --nolock
echo "END $(date -Is)"
