#!/bin/bash
# Rebuild every figure's prepared tables into results/figure_data/.
#
# Each script here reads raw run outputs from results/outputs/ (or runs its own
# simulation/optimization) and writes the tidy tables one figure draws from. The
# plotting scripts in results/figures/plotting_scripts/ then only read those
# tables -- see results/figure_data/README.md.
#
# A figure whose run has not been produced on this machine fails with a
# "missing run: <path>" message. That is expected, so this driver keeps going and
# prints a summary at the end; it exits non-zero only if something failed.
#
# Self-anchoring: run it from anywhere.
set -uo pipefail

cd "$(dirname "$0")"

REPO_ROOT="$(cd ../../.. && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Prep scripts import from src/ and utils/snakemake_utils/ via _common.add_src_to_path,
# so only the repo root has to be on PYTHONPATH here.
PYTHON="${PYTHON:-python}"

succeeded=()
failed=()

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

echo
echo "Prepared ${#succeeded[@]} figure(s):"
for name in "${succeeded[@]}"; do echo "  ok      $name"; done
if (( ${#failed[@]} )); then
    echo "Failed ${#failed[@]} figure(s):"
    for name in "${failed[@]}"; do echo "  FAILED  $name"; done
    echo
    echo "A 'missing run' failure means that figure's pipeline run is not on this"
    echo "machine; produce it with the matching config under data/configs/ first."
    exit 1
fi
echo "Figure data written to results/figure_data/"
