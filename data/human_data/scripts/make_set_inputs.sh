#!/usr/bin/env bash
# Build one sample set's singer-snakemake input directory.
#
# Called once per set by subset.sh (locally via xargs, or as one task of a SLURM array).
# Writes SingerSnakemake_SampleSets/<set>/ containing, per chromosome:
#
#   chr{N}.vcf.gz(.tbi)      subset to this set's samples, re-trimmed to segregating sites
#   chr{N}.mask.bed          symlink to the shared inaccessible-interval mask
#   chr{N}.hapmap            symlink to the shared recombination map
#   chr{N}.ancestral.fa.gz   symlink to the shared ancestral FASTA
#   chr{N}.meta.csv          sample metadata IN VCF ORDER (see write_set_metadata.py)
#
# singer-snakemake locates every non-VCF input by string substitution on the VCF path, so all
# five must share a stem and sit in one directory. The three shared files are identical across
# sets, hence symlinks rather than 8 copies of ~700 MB of ancestral FASTA.
#
# Usage: make_set_inputs.sh <set_name> <human_data_dir>
set -euo pipefail

SET=$1
ROOT=$2

command -v bcftools >/dev/null || {
    echo "[${SET}] bcftools not on PATH -- activate CoaSTT_env first" >&2
    exit 1
}

SETS_DIR="${ROOT}/resources/hgdp/sets"
STATPHASE="${ROOT}/resources/hgdp/statphase"
GRCH38="${ROOT}/resources/grch38"
OUT="${ROOT}/SingerSnakemake_SampleSets/${SET}"
SENTINEL="${ROOT}/SingerSnakemake_SampleSets/${SET}/.complete"

SAMPLES="${SETS_DIR}/${SET}.csv"
[[ -f "$SAMPLES" ]] || { echo "[${SET}] no sample list at ${SAMPLES}" >&2; exit 1; }

if [[ -f "$SENTINEL" ]]; then
    echo "[${SET}] already complete, skipping"
    exit 0
fi

mkdir -p "$OUT"
n_samples=$(wc -l < "$SAMPLES")
echo "[${SET}] building input dir for ${n_samples} samples"

for n in $(seq 1 22); do
    chrom="chr${n}"
    vcf="${OUT}/${chrom}.vcf.gz"

    if [[ -f "${vcf}.tbi" && "${vcf}.tbi" -nt "$vcf" ]]; then
        echo "[${SET}] ${chrom}: vcf present, skipping"
    else
        # -a trims alleles no longer carried by anyone in the subset; -c 1 then drops sites
        # that have become monomorphic. Both matter: the workflow computes its filters after
        # our subsetting, and stale ALT alleles would look multiallelic.
        bcftools view \
            -S "$SAMPLES" \
            -a -c 1 \
            -v snps \
            -Oz -o "$vcf" \
            "${STATPHASE}/${chrom}.bcf"
        bcftools index -t -f "$vcf"
        echo "[${SET}] ${chrom}: $(bcftools index -n "$vcf") sites"
    fi

    # Shared resources: relative symlinks so the tree stays relocatable. Two levels up from
    # SingerSnakemake_SampleSets/<SET>/ is data/human_data/, which is where resources/ lives.
    ln -sfn "../../resources/grch38/mask/${chrom}.mask.bed" "${OUT}/${chrom}.mask.bed"
    ln -sfn "../../resources/grch38/hapmap/${chrom}.hapmap" "${OUT}/${chrom}.hapmap"
    ln -sfn "../../resources/grch38/ancestral/${chrom}.ancestral.fa.gz" \
        "${OUT}/${chrom}.ancestral.fa.gz"
done

# Per-sample missing-genotype masks: lightly-missing sites are marked so singer-snakemake
# imputes them rather than dropping them outright.
echo "[${SET}] writing per-sample missing-genotype masks"
python "${ROOT}/scripts/make_sample_masks.py" --input-dir "$OUT"

# Metadata must be written last: it is ordered against the VCF sample order.
python "${ROOT}/scripts/write_set_metadata.py" \
    --metadata "${ROOT}/resources/hgdp/metadata.tsv" \
    --input-dir "$OUT"

touch "$SENTINEL"
echo "[${SET}] done"
