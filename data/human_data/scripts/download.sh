#!/usr/bin/env bash
# Download every raw input and derive the shared per-chromosome resources.
#
#   bash download.sh [--threads N]
#
# Idempotent: each step is skipped when its DERIVED artefact already exists, so an interrupted
# run resumes without re-deriving finished work. Raw downloads resume mid-file (curl -C -).
#
# Produces:
#   resources/raw/                     downloads exactly as fetched
#   resources/grch38/chrom.sizes       derived from the ancestral FASTAs (source of truth)
#   resources/grch38/ancestral/        chr{1..22}.ancestral.fa.gz
#   resources/grch38/mask/             chr{1..22}.mask.bed  (INACCESSIBLE intervals)
#   resources/grch38/hapmap/           chr{1..22}.hapmap
#   resources/hgdp/statphase/          chr{1..22}.bcf(.csi), GT only
#   resources/hgdp/metadata.tsv
#   resources/hgdp/sets/               the 8 sample set definitions
#
# Run `bash subset.sh` afterwards to build the per-set singer-snakemake input directories.
set -euo pipefail

THREADS=4
while [[ $# -gt 0 ]]; do
    case $1 in
        --threads) THREADS=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${ROOT}/resources/raw"
GRCH38="${ROOT}/resources/grch38"
HGDP="${ROOT}/resources/hgdp"
SCRATCH="${ROOT}/.scratch"

mkdir -p "$RAW" "$GRCH38"/{mask,hapmap,ancestral} "$HGDP"/{statphase,sets} "$SCRATCH"

for tool in bcftools bedtools bgzip python; do
    command -v "$tool" >/dev/null || {
        echo "ERROR: $tool is not on PATH. Run 'conda activate CoaSTT_env' first." >&2
        exit 1
    }
done

SANGER="https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516"
ENSEMBL="https://ftp.ensembl.org/pub/release-113/fasta/ancestral_alleles"
ENSEMBL_MIRROR="https://ftp.ebi.ac.uk/pub/ensembl/release-113/fasta/ancestral_alleles"

fetch () {  # fetch <url> <dest> [fallback_url]
    local url=$1 dest=$2 fallback=${3:-}
    if [[ -s "$dest" ]]; then
        echo "  [skip] $(basename "$dest") already downloaded"
        return 0
    fi
    echo "  [get ] $(basename "$dest")"
    if ! curl -fSL --retry 3 --retry-delay 10 -C - -o "$dest" "$url"; then
        if [[ -n "$fallback" ]]; then
            echo "  [warn] primary host failed, trying fallback" >&2
            curl -fSL --retry 3 --retry-delay 10 -C - -o "$dest" "$fallback"
        else
            return 1
        fi
    fi
}

echo "== 1/7 downloading raw inputs =="

fetch "${SANGER}/statphase/hgdp_wgs.20190516.statphase.autosomes.vcf.gz" \
      "${RAW}/hgdp_wgs.20190516.statphase.autosomes.vcf.gz"
fetch "${SANGER}/statphase/hgdp_wgs.20190516.statphase.autosomes.vcf.gz.tbi" \
      "${RAW}/hgdp_wgs.20190516.statphase.autosomes.vcf.gz.tbi"
fetch "${SANGER}/accessibility-mask/hgdp_wgs.20190516.mask.bed" \
      "${RAW}/hgdp_wgs.20190516.mask.bed"
fetch "${SANGER}/accessibility-mask/README.hgdp_wgs.20190516.mask.txt" \
      "${RAW}/README.hgdp_wgs.20190516.mask.txt"
fetch "${SANGER}/metadata/hgdp_wgs.20190516.metadata.txt" \
      "${RAW}/hgdp_wgs.20190516.metadata.txt"
fetch "${ENSEMBL}/homo_sapiens_ancestor_GRCh38.tar.gz" \
      "${RAW}/homo_sapiens_ancestor_GRCh38.tar.gz" \
      "${ENSEMBL_MIRROR}/homo_sapiens_ancestor_GRCh38.tar.gz"

# --- deCODE 2019 sex-averaged recombination map ------------------------------------------
# Science puts its supplements behind bot protection: as of 2026-09-11 all three known URLs
# return HTTP 403 to curl even with a browser User-Agent, so this cannot be automated. The
# file was fetched manually through a browser and placed at
# resources/raw/aau1043_datas3.gz (gzip it if your download is uncompressed).
#
# Re-enable the line below if Science ever drops the block; the rest of the script already
# handles the file identically whichever way it arrives.
#
# fetch "https://www.science.org/doi/suppl/10.1126/science.aau1043/suppl_file/aau1043_datas3.gz" \
#       "${RAW}/aau1043_datas3.gz" \
#       "https://science.sciencemag.org/highwire/filestream/721792/field_highwire_adjunct_files/4/aau1043_DataS3.gz"

if [[ ! -s "${RAW}/aau1043_datas3.gz" ]]; then
    cat >&2 <<'EOF'

ERROR: the deCODE 2019 recombination map is missing.

  Expected at: resources/raw/aau1043_datas3.gz

It cannot be downloaded automatically -- science.org returns HTTP 403 to non-browser clients.
Download it by hand from:

  https://www.science.org/doi/10.1126/science.aau1043
  (supplementary materials, "Data S3")

then place it at the path above, gzipping it first if it arrives uncompressed.
EOF
    exit 1
fi

echo "== 2/7 verifying the statphase VCF against its index =="
VCF="${RAW}/hgdp_wgs.20190516.statphase.autosomes.vcf.gz"
n_records=$(bcftools index -n "${VCF}.tbi" 2>/dev/null)
n_samples=$(bcftools query -l "$VCF" 2>/dev/null | wc -l)
echo "  ${n_records} records, ${n_samples} samples"
[[ "$n_samples" -eq 929 ]] || { echo "ERROR: expected 929 samples, got ${n_samples}" >&2; exit 1; }

echo "== 3/7 ancestral FASTAs + chrom.sizes =="
if [[ -s "${GRCH38}/chrom.sizes" ]]; then
    echo "  [skip] chrom.sizes exists"
else
    python "${ROOT}/scripts/make_ancestral.py" \
        --tarball "${RAW}/homo_sapiens_ancestor_GRCh38.tar.gz" \
        --outdir "${GRCH38}/ancestral" \
        --chrom-sizes "${GRCH38}/chrom.sizes" \
        --workdir "${SCRATCH}/ancestral"
fi

echo "== 4/7 accessibility mask -> inaccessible intervals =="
if [[ -s "${GRCH38}/mask/chr22.mask.bed" ]]; then
    echo "  [skip] masks exist"
else
    python "${ROOT}/scripts/make_masks.py" \
        --mask "${RAW}/hgdp_wgs.20190516.mask.bed" \
        --chrom-sizes "${GRCH38}/chrom.sizes" \
        --outdir "${GRCH38}/mask"
fi

echo "== 5/7 deCODE map -> per-chromosome hapmaps =="
if [[ -s "${GRCH38}/hapmap/chr22.hapmap" ]]; then
    echo "  [skip] hapmaps exist"
else
    python "${ROOT}/scripts/make_hapmaps.py" \
        --decode "${RAW}/aau1043_datas3.gz" \
        --chrom-sizes "${GRCH38}/chrom.sizes" \
        --outdir "${GRCH38}/hapmap"
fi

echo "== 6/7 metadata + sample sets =="
if [[ -s "${HGDP}/metadata.tsv" ]]; then
    echo "  [skip] metadata.tsv exists"
else
    python "${ROOT}/scripts/make_metadata.py" \
        --metadata "${RAW}/hgdp_wgs.20190516.metadata.txt" \
        --out "${HGDP}/metadata.tsv"
fi
if [[ -s "${HGDP}/sets/ALL.csv" ]]; then
    echo "  [skip] sample sets exist"
else
    python "${ROOT}/scripts/make_sample_sets.py" \
        --metadata "${HGDP}/metadata.tsv" \
        --outdir "${HGDP}/sets"
fi

# The palette needs only metadata, so it can be built here. The per-set pop_labels.csv
# cannot: its row order is the VCF's, which does not exist until subset.sh has run. Build
# those with make_labels.sh once the set inputs are in place.
python "${ROOT}/scripts/HGDP_color_palette.py" \
    --metadata "${HGDP}/metadata.tsv" \
    --outdir "${ROOT}/resources/population_labels_colors"

echo "== 7/7 splitting statphase into per-chromosome GT-only BCFs =="
bash "${ROOT}/scripts/split_statphase.sh" "$VCF" "${HGDP}/statphase" "$THREADS"

echo
echo "== verifying derived resources =="
python "${ROOT}/scripts/verify_resources.py" --resources "${ROOT}/resources"

rmdir "${SCRATCH}/ancestral" "$SCRATCH" 2>/dev/null || true

echo
echo "Done. Next: bash subset.sh"
