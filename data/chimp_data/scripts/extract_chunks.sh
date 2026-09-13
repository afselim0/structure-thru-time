#!/usr/bin/env bash
# Cut the final posterior draw of each chromosome into per-chunk tree sequences for CoaSTT.
#
#   bash scripts/extract_chunks.sh                 # the production run
#   bash scripts/extract_chunks.sh chimp_pilot     # a pilot run, or any other results/<name>
#
# Writes results/<name>/ts_chunks/<chrom>_<start>_<end>.trees -- FLAT, with chunks from every
# chromosome together and no per-chromosome subdirectories. That is what CountCoalEvents
# wants: it globs *.trees and turns each filename into a contig id
# (src/CountCoalEvents.py:141-151).
#
# The chimpanzee panel is a single sample set, so `chimp` is the only name under results/;
# the human side has one per set because it has eight. The extraction script itself is
# shared -- ../human_data/scripts/extract_ts_chunks.py -- rather than duplicated, since
# nothing in it is human-specific.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

NAME="${1:-chimp}"
SRC="${ROOT}/results/${NAME}/singer-snakemake_outputs"
OUT="${ROOT}/results/${NAME}/ts_chunks"
[[ -d "$SRC" ]] || { echo "no singer output at $SRC" >&2; exit 1; }

# --- gate: every chromosome must contribute the SAME replicate -------------------------
#
# extract_ts_chunks.py takes the highest-numbered replicate per chromosome, which is correct
# only if replicates accumulate in order. A crashed or OOM-killed merge_chunks breaks that:
# the rule is parameterised per replicate and the jobs run concurrently, so a kill removes
# whichever were in flight -- not necessarily the last ones. On the first genome-wide run
# chr8 came back missing replicates 1, 6 AND 15 while keeping 14, so it would have silently
# contributed a different, less burnt-in draw than every other chromosome.
#
# That is invisible downstream: the file is valid, the chunk count is right, and CoaSTT has
# no way to know one contig came from a different point in the chain. So refuse to build a
# mixed dataset, and say exactly which chromosome to re-merge.
declare -A TOP
MAXTOP=-1
for d in "${SRC}"/*/; do
    c=$(basename "$d")
    [[ -d "${d}trees" ]] || continue
    top=$(ls "${d}trees/${c}."*.tsz 2>/dev/null \
          | sed "s|.*/${c}\.\([0-9]*\)\.tsz|\1|" | sort -n | tail -1)
    [[ -n "$top" ]] || { echo "  $c: no .tsz replicates yet" >&2; continue; }
    TOP[$c]=$top
    (( top > MAXTOP )) && MAXTOP=$top
done
[[ ${#TOP[@]} -gt 0 ]] || { echo "no finished chromosomes under $SRC" >&2; exit 1; }

BEHIND=()
for c in "${!TOP[@]}"; do
    (( TOP[$c] < MAXTOP )) && BEHIND+=("$c (has ${TOP[$c]}, expected ${MAXTOP})")
done
if [[ ${#BEHIND[@]} -gt 0 ]]; then
    echo "REFUSING to build a mixed-replicate dataset." >&2
    echo "These chromosomes are missing replicate ${MAXTOP}:" >&2
    printf '  %s\n' "${BEHIND[@]}" >&2
    echo >&2
    echo "Re-run the merge step for them -- the SINGER output survives on disk, so this" >&2
    echo "costs minutes, not the hours of inference:" >&2
    echo "  bash scripts/run_singer.sh" >&2
    exit 1
fi
echo "All ${#TOP[@]} chromosomes at replicate ${MAXTOP}."

python "${ROOT}/../human_data/scripts/extract_ts_chunks.py" \
    --singer-output "$SRC" \
    --out-dir "$OUT"
