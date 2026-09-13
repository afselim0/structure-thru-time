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
# Silences a harmless "Can't find a usable init.tcl" from this env's Tcl, which was
# built for a differently-named prefix and is found via the lib path above.
export TCL_LIBRARY="/project/jnovembre/afselim/CoaSTT_env/lib/tcl8.6"

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
        --out-dir "${ROOT}/results/${s}/ts_chunks" \
        --config "${ROOT}/drivers/singer_config.yaml"

    # Now that extraction has SUCCEEDED, the pre-convergence draws are dead weight:
    # merge_chunks writes all 100, of which only the last 20 are past burn-in. Report what
    # could be freed, but do not delete it automatically -- reclaiming disk is not worth doing
    # behind someone's back, and the dry run prints the exact command.
    echo
    python "${ROOT}/scripts/prune_burnin_draws.py" \
        --singer-output "$out" \
        --config "${ROOT}/drivers/singer_config.yaml" \
        --ts-chunks "${ROOT}/results/${s}/ts_chunks" || true
    echo "To reclaim that space, re-run the command above with --apply"
done
