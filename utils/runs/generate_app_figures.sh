#!/bin/bash
# Regenerate every application (empirical: human / chimp) figure into results/figures/app_figs/.
# Each script is self-anchoring (it chdir's to its own directory), so this can be
# run from anywhere. PNG intermediates are written to results/figures/app_figs_png/.
#
# These scripts only plot: they read prepared tables from results/figure_data/.
# To rebuild those tables from the raw run outputs in results/outputs/, run
#   bash utils/runs/prepare_all_figure_data.sh
# (or the single script utils/runs/prepare_figure_data/<figure_name>.py).
# See results/figure_data/README.md.
set -euo pipefail

# The figure scripts are self-anchoring but write their output relative to
# results/figures/, so run from there rather than from this driver's own
# directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/results/figures"

# Every figure script does `from results.figures.plotting_style import *`, which
# resolves as a namespace package only with the repo root on the import path.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT_DIR="./plotting_scripts"
APP_SCRIPTS="mean_coal_rates.py \
              chimp_dataset.py \
              split_and_sawtooth_singer.py \
              complex_4_pops_singer.py \
              double_SS_2d_singer.py \
              double_admix_GNN_singer.py"

mkdir -p app_figs app_figs_png

for name in $APP_SCRIPTS; do
    script="$SCRIPT_DIR/$name"
    if [[ ! -f "$script" ]]; then
        echo "Expected application figure script not found: $script" >&2
        exit 1
    fi
    echo "Running $name..."
    python "$script"
done

echo "Application figures written to results/figures/app_figs/"
