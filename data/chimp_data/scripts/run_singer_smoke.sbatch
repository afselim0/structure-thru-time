#!/bin/bash
#SBATCH --job-name=chimp_singer21
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
# CoaSTT_env/lib must precede system libs or snakemake hits a libstdc++ mismatch on midway3.
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR/../singer-snakemake"
snakemake --configfile ../chimp_data/singer_config_smoke.yaml \
          --cores "$SLURM_CPUS_PER_TASK" --rerun-incomplete --nolock
echo "SMOKE RUN DONE"
