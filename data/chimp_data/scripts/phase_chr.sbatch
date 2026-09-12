#!/bin/bash
#SBATCH --job-name=phase_chimp
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

# SHAPEIT5 phase_common on one whole chromosome. Runs under SLURM because the login session
# is capped at an 8 GiB cgroup. Parameters are fixed, not tuned: --hmm-ne 10000 follows
# De Manuel et al. 2016 and was measured insensitive over 10,000-30,000; no reference panel
# (the subspecies have none); no genetic map (at n=54 the map is not the limiting factor).
set -euo pipefail

CHROM=${1:?usage: sbatch phase_chr.sbatch chrN}
export PATH="/project/jnovembre/afselim/shapeit_env/bin:/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p work/phased logs

# Site-filtered, UNPHASED input. Order is filter -> phase, never the reverse: SHAPEIT5
# emits GT only, stripping AD/DP, so the depth and allelic-imbalance filters cannot be
# computed once phasing has run.
IN=work/filtered/${CHROM}.sitefiltered.vcf.gz
OUT=work/phased/${CHROM}.vcf.gz
LEN=$(awk -v c="$CHROM" '$1==c{print $2}' resources/panTro6/chrom.sizes)

SHAPEIT5_phase_common \
  --input "$IN" \
  --output work/phased/${CHROM}.bcf \
  --region "${CHROM}:1-${LEN}" \
  --hmm-ne 10000 \
  --thread "$SLURM_CPUS_PER_TASK"

bcftools view -Oz -o "$OUT" work/phased/${CHROM}.bcf
bcftools index -t "$OUT"

# SHAPEIT5 drops NON-BIALLELIC records by design -- it phases biallelic sites only. So the
# right invariant is not "no sites lost" (measured wrong: 30,811 multiallelics went missing
# on chr21) but "every BIALLELIC site survived, and nothing else was touched".
n_in=$(bcftools index -n "$IN")
n_out=$(bcftools index -n "$OUT")
# BIALLELIC OF ANY TYPE, not biallelic SNPs: SHAPEIT5 keeps biallelic spanning deletions
# and indels too (12,723 of them on chr21). Measured, not assumed -- the output position set
# equals `-m2 -M2` exactly.
n_bi=$(bcftools view -H "$IN" -m2 -M2 | wc -l)
echo "input=$n_in biallelic_in=$n_bi output=$n_out"
if [ "$n_out" -ne "$n_bi" ]; then
  echo "ERROR: phased output has $n_out sites, expected the $n_bi biallelic records" >&2
  exit 1
fi
# Every output genotype must be phased; an unphased '/' means SHAPEIT5 left work undone.
n_unphased=$(bcftools query -f '[%GT\n]' "$OUT" | grep -c '/' || true)
if [ "$n_unphased" -ne 0 ]; then
  echo "ERROR: $n_unphased unphased genotypes in the output" >&2
  exit 1
fi
echo "all $n_bi biallelic records phased; $((n_in - n_bi)) multiallelic dropped by SHAPEIT5"
rm -f work/phased/${CHROM}.bcf work/phased/${CHROM}.bcf.csi
echo "phased $CHROM OK"
