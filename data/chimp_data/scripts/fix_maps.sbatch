#!/bin/bash
#SBATCH --job-name=chimp_fixmap
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
cd "$SLURM_SUBMIT_DIR"
CH=(chr2A chr2B); CHROM=${CH[$SLURM_ARRAY_TASK_ID]}
python3 scripts/build_hapmap.py "$CHROM"
python3 scripts/build_ancestral_fasta.py "$CHROM"
echo "fixmap $CHROM OK"
