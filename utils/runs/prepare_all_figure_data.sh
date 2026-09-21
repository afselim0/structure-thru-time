#!/bin/bash
# Rebuild every figure's prepared tables into results/figure_data/.
#
#     bash prepare_all_figure_data.sh
#
# Each script in prepare_figure_data/ reads raw run outputs from
# results/outputs/ (or runs its own simulation/optimization) and writes the tidy
# tables one figure draws from. The plotting scripts in
# results/figures/plotting_scripts/ then only read those tables -- see
# results/figure_data/README.md.
#
# This stage always runs here, never on SLURM: the scripts are light next to the
# CoaSTT runs that produced their inputs, so there is nothing to schedule. Run
# generate_sim_results.sh and generate_app_results.sh first.
#
# A figure whose run has not been produced on this machine fails with a
# "missing run: <path>" message. That is expected, so this driver keeps going,
# writes a per-script report to figure_data.txt beside this script, and exits
# non-zero if anything failed.
#
# Self-anchoring: run it from anywhere.
set -uo pipefail

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$RUNS_DIR/prepare_figure_data"
REPORT="$RUNS_DIR/figure_data.txt"

REPO_ROOT="$(cd "$RUNS_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The prep scripts import from src/ and utils/snakemake_utils/ via
# _common.add_src_to_path, so only the repo root has to be on PYTHONPATH here.
# They are run from their own directory, as they were when they lived beside
# this driver, so any relative path inside one still resolves.
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

succeeded=()
failed=()

shopt -s nullglob
for script in *.py; do
    # _common.py is the shared helper module, not a figure
    [[ "$script" == "_common.py" ]] && continue
    name="${script%.py}"
    echo "=== $name ==="
    if "$PYTHON" "$script"; then
        succeeded+=("$name")
    else
        failed+=("$name")
        echo "  !! $name failed (see the message above)"
    fi
done

# The report: the same summary that goes to the terminal, kept as a file so a
# long run can be checked over afterwards. Rewritten from scratch each time, so
# it always describes the run that just finished.
{
    echo "Figure data prepared $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Repository: $REPO_ROOT"
    echo
    echo "Succeeded (${#succeeded[@]}):"
    for name in "${succeeded[@]}"; do echo "  ok      $name"; done
    if (( ${#failed[@]} )); then
        echo
        echo "Failed (${#failed[@]}):"
        for name in "${failed[@]}"; do echo "  FAILED  $name"; done
        echo
        echo "A 'missing run' failure means that figure's pipeline run is not on"
        echo "this machine; produce it with the matching config under"
        echo "data/configs/ first."
    fi
} > "$REPORT"

echo
cat "$REPORT"
echo
echo "Report written to $REPORT"

if (( ${#failed[@]} )); then
    exit 1
fi
echo "Figure data written to results/figure_data/"
