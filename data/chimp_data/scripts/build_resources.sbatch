#!/bin/bash
#SBATCH --job-name=chimp_res
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=1-23
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
cd "$SLURM_SUBMIT_DIR"
CHROM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" resources/panTro6/chrom.sizes | cut -f1)

# Recombination map: Stevison ellioti, hg18 -> hg38 -> panTro6, one global scalar.
if [ ! -s resources/panTro6/hapmap/${CHROM}.hapmap ]; then
  python3 scripts/build_hapmap.py "$CHROM"
fi
# Ancestral state: hg38 projected via the panTro6ToHg38 chain, with the per-strand gate.
if [ ! -s resources/panTro6/ancestral/${CHROM}.ancestral.fa.gz ]; then
  python3 scripts/build_ancestral_fasta.py "$CHROM"
fi
echo "resources $CHROM OK"
