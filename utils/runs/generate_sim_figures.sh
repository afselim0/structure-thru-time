#!/bin/bash
# Regenerate every simulation figure into results/figures/sim_figs/.
# Each script is self-anchoring (it chdir's to its own directory), so this can be
# run from anywhere. PNG intermediates are written to results/figures/sim_figs_png/.
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
APP_SCRIPTS="mean_coal_rates.py chimp_dataset.py"   # empirical figures — see generate_app_figures.sh

mkdir -p sim_figs sim_figs_png

for script in "$SCRIPT_DIR"/*.py; do
    name=$(basename "$script")
    case " $APP_SCRIPTS " in *" $name "*) continue ;; esac
    echo "Running $name..."
    python "$script"
done

echo "Simulation figures written to results/figures/sim_figs/"
