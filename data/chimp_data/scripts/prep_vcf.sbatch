#!/bin/bash
#SBATCH --job-name=chimp_vcf
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=1-23
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
cd "$SLURM_SUBMIT_DIR"

CHROM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" resources/panTro6/chrom.sizes | cut -f1)
ACC=$(awk -v c="$CHROM" '$2==c{print $1}' resources/panTro6/accession_to_chrom.txt)
echo "=== $CHROM ($ACC) ==="

RAW=work/filtered/${CHROM}.raw.vcf.gz
REN=work/filtered/${CHROM}.renamed.vcf.gz

# Split to this chromosome, drop bonobos, drop sites no chimp carries.
if [ ! -s "$RAW" ]; then
  bcftools view -S resources/samples/chimps.txt -a -r "$ACC" resources/vcf/full.vcf.gz -Ou \
    | bcftools view -c 1 -Oz -o "$RAW"
  bcftools index -t "$RAW"
fi
echo "after bonobo removal: $(bcftools index -n $RAW)"

# Rename the GenBank accession to chrN so every downstream file shares one contig name.
if [ ! -s "$REN" ]; then
  printf '%s\t%s\n' "$ACC" "$CHROM" > work/rename_${CHROM}.txt
  bcftools annotate --rename-chrs work/rename_${CHROM}.txt "$RAW" -Oz -o "$REN"
  bcftools index -t "$REN"
fi

# Site filters (GATK + allelic imbalance + depth) and filter.txt.
python3 scripts/filter_sites.py "$CHROM"
echo "prep_vcf $CHROM OK"
