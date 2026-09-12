#!/bin/bash
#SBATCH --job-name=chimp_phase
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-23
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/shapeit_env/bin:/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
cd "$SLURM_SUBMIT_DIR"

CHROM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" resources/panTro6/chrom.sizes | cut -f1)
LEN=$(awk -v c="$CHROM" '$1==c{print $2}' resources/panTro6/chrom.sizes)
IN=work/filtered/${CHROM}.sitefiltered.vcf.gz
OUT=work/phased/${CHROM}.vcf.gz
[ -s "$OUT" ] && { echo "$CHROM already phased"; exit 0; }

SHAPEIT5_phase_common --input "$IN" --output work/phased/${CHROM}.bcf \
  --region "${CHROM}:1-${LEN}" --hmm-ne 10000 --thread "$SLURM_CPUS_PER_TASK"
bcftools view -Oz -o "$OUT" work/phased/${CHROM}.bcf
bcftools index -t "$OUT"

n_bi=$(bcftools view -H "$IN" -m2 -M2 | wc -l)
n_out=$(bcftools index -n "$OUT")
[ "$n_out" -eq "$n_bi" ] || { echo "ERROR: $n_out != $n_bi biallelic" >&2; exit 1; }
n_unphased=$(bcftools query -f '[%GT\n]' "$OUT" | grep -c '/' || true)
[ "$n_unphased" -eq 0 ] || { echo "ERROR: $n_unphased unphased" >&2; exit 1; }
rm -f work/phased/${CHROM}.bcf work/phased/${CHROM}.bcf.csi
echo "phase $CHROM OK: $n_out biallelic phased"
