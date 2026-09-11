#!/usr/bin/env bash
# Split the genome-wide statphase VCF into per-chromosome GT-only BCFs.
#
# This is the expensive hoist: done once here, the 8 per-set subset jobs each read a small
# indexed BCF for one chromosome instead of streaming 6.3 GB of genome-wide VCF. Stripping
# everything but FORMAT/GT is most of the size win, and nothing downstream reads INFO --
# singer-snakemake's `allel.read_vcf` takes CHROM/POS/ID/REF/ALT/GT only.
#
# Indels are dropped here rather than during subsetting. singer-snakemake never inspects
# REF/ALT strings, so an indel would be silently treated as a SNP; `-v snps` is the only
# place that is caught.
#
# Usage: split_statphase.sh <in.vcf.gz> <outdir> [threads]
set -euo pipefail

command -v bcftools >/dev/null || {
    echo "[split_statphase] bcftools not on PATH -- activate CoaSTT_env first" >&2
    exit 1
}

VCF=$1
OUTDIR=$2
THREADS=${3:-4}

mkdir -p "$OUTDIR"

for n in $(seq 1 22); do
    chrom="chr${n}"
    out="${OUTDIR}/${chrom}.bcf"

    if [[ -f "${out}.csi" && "${out}.csi" -nt "$out" ]]; then
        echo "[split_statphase] ${chrom}: already done, skipping"
        continue
    fi

    echo "[split_statphase] ${chrom}: extracting"
    bcftools view \
        --threads "$THREADS" \
        -r "$chrom" \
        -v snps \
        -Ou "$VCF" \
    | bcftools annotate \
        --threads "$THREADS" \
        -x "INFO,^FORMAT/GT" \
        -Ob -o "$out"

    bcftools index --threads "$THREADS" -f "$out"
    echo "[split_statphase] ${chrom}: $(bcftools index -n "$out") records"
done

echo "[split_statphase] done"
