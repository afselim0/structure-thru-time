#!/usr/bin/env bash
# Build the colour palette and the per-set pop_labels.csv / pop_colors.csv for CoaSTT.
#
# Writes resources/population_labels_colors/:
#
#   global_palette.{csv,png}    region hue + population ramp; used for ALL
#   regional_palette.{csv,png}  discrete within each region; used for the 7 region sets
#   <SET>_pop_labels.csv        one row per haplotype, in tree-sequence sample order
#   <SET>_pop_colors.csv        population -> colour for that set
#
# Run after subset.sh, which writes the chr1.meta.csv files that fix the sample order.
# If the tree sequences exist too, the order is additionally asserted against them.
#
# Usage: make_labels.sh [human_data_dir]
set -euo pipefail

ROOT=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
RESOURCES="${ROOT}/resources"
LABELS="${RESOURCES}/population_labels_colors"

python "${ROOT}/scripts/HGDP_color_palette.py" \
    --metadata "${RESOURCES}/hgdp/metadata.tsv" \
    --outdir "$LABELS"

python "${ROOT}/scripts/make_sample_sets.py" \
    --metadata "${RESOURCES}/hgdp/metadata.tsv" \
    --outdir "${RESOURCES}/hgdp/sets" \
    --labels-outdir "$LABELS" \
    --sets-dir "${ROOT}/SingerSnakemake_SampleSets" \
    --trees-dir "${ROOT}/results"
