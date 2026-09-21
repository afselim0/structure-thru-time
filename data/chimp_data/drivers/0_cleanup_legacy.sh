#!/usr/bin/env bash
# ONE-OFF -- remove the leftovers of the first genome-wide run.
#
#   bash drivers/0_cleanup_legacy.sh
#
# Not part of the pipeline. The first genome-wide attempt used an sbatch array with a
# per-chromosome snakemake workdir under runs/work/, and wrote its output to
# results/singer-snakemake_outputs/ rather than results/chimp/. Both layouts are superseded:
# run state now goes to scratch OUTSIDE the repo (see scripts/run_singer.sh) and output goes
# to results/<name>/.
#
# REFUSES to run while any chimp_singer job is still in the queue, because those jobs are
# still writing to both locations.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

live=$(squeue -u "$USER" -h -n chimp_singer 2>/dev/null | wc -l)
if [[ "$live" -gt 0 ]]; then
    echo "REFUSING: ${live} chimp_singer job(s) still in the queue." >&2
    echo "They are still writing to runs/work/ and results/singer-snakemake_outputs/." >&2
    echo "Wait for them to finish, then re-run this." >&2
    exit 1
fi

if [[ -d results/singer-snakemake_outputs ]]; then
    mkdir -p results/chimp
    echo "moving results/singer-snakemake_outputs -> results/chimp/"
    mv results/singer-snakemake_outputs results/chimp/singer-snakemake_outputs
fi

if [[ -d runs ]]; then
    echo "removing runs/ (regenerable snakemake state)"
    rm -rf runs
fi

echo "Done. Layout is now:"
echo "  results/chimp/singer-snakemake_outputs/   SINGER output"
echo "  results/chimp/ts_chunks/draw_<rep>/        CoaSTT input (step 9)"
