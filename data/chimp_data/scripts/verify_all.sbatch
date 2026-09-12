#!/bin/bash
#SBATCH --job-name=chimp_verify
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-23
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
cd "$SLURM_SUBMIT_DIR"
CHROM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" resources/panTro6/chrom.sizes | cut -f1)
python3 scripts/verify_inputs.py "$CHROM"
