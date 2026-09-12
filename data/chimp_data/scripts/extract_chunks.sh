#!/usr/bin/env bash
# Cut the final posterior draw of each chromosome into per-chunk tree sequences for CoaSTT.
#
#   bash scripts/extract_chunks.sh                        # the production run
#   bash scripts/extract_chunks.sh results/chr21_smoke     # any other output tree
#
# Writes results/ts_chunks/<chrom>_<start>_<end>.trees -- FLAT, with chunks from every
# chromosome together and no per-chromosome subdirectories. That is what CountCoalEvents
# wants: it globs *.trees and turns each filename into a contig id
# (src/CountCoalEvents.py:141-151).
#
# The chimpanzee panel is a single sample set, so there is no per-set level here; the human
# side has results/<SET>/ts_chunks because it has eight. The extraction script itself is
# shared -- ../human_data/scripts/extract_ts_chunks.py -- rather than duplicated, since
# nothing in it is human-specific.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

SRC="${1:-${ROOT}/results/singer-snakemake_outputs}"
[[ "$SRC" = /* ]] || SRC="${ROOT}/${SRC#./}"
OUT="${2:-${ROOT}/results/ts_chunks}"
[[ "$OUT" = /* ]] || OUT="${ROOT}/${OUT#./}"

# Accept either the singer output directory itself or the run root containing it.
[[ -d "${SRC}/singer-snakemake_outputs" ]] && SRC="${SRC}/singer-snakemake_outputs"
[[ -d "$SRC" ]] || { echo "no singer output at $SRC" >&2; exit 1; }

python "${ROOT}/../human_data/scripts/extract_ts_chunks.py" \
    --singer-output "$SRC" \
    --out-dir "$OUT"
