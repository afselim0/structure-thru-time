#!/usr/bin/env bash
# Cut each finished set's final posterior draw into per-chunk tree sequences for CoaSTT.
#
#   bash scripts/extract_chunks.sh            # every set with singer-snakemake output
#   bash scripts/extract_chunks.sh OCEANIA    # just one
#
# Writes results/<SET>/ts_chunks/<chr>_<start>_<end>.trees -- flat, no chromosome
# subdirectories, which is what CountCoalEvents expects (it globs *.trees and treats each
# filename as a contig id).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

SETS=("$@")
if [[ ${#SETS[@]} -eq 0 ]]; then
    mapfile -t SETS < <(
        cd "${ROOT}/results" 2>/dev/null && \
        for d in */; do [[ -d "${d}singer-snakemake_outputs" ]] && echo "${d%/}"; done
    )
fi
[[ ${#SETS[@]} -gt 0 ]] || { echo "no sets with singer-snakemake output found" >&2; exit 1; }

for s in "${SETS[@]}"; do
    out="${ROOT}/results/${s}/singer-snakemake_outputs"
    [[ -d "$out" ]] || { echo "[${s}] no output directory, skipping" >&2; continue; }
    echo "[${s}]"
    python "${ROOT}/scripts/extract_ts_chunks.py" \
        --singer-output "$out" \
        --out-dir "${ROOT}/results/${s}/ts_chunks"
done
