#!/bin/bash
# Regenerate every simulation figure into sim_figs/.
# Each script is self-anchoring (it chdir's to its own directory), so this can be
# run from anywhere. PNG intermediates are written to sim_figs_png/.
#
# These scripts only plot: they read prepared tables from results/figure_data/.
# To rebuild those tables from the raw run outputs in results/outputs/, run
#   bash utils/runs/prepare_figure_data/prepare_all_figure_data.sh
# (or the single script utils/runs/prepare_figure_data/<figure_name>.py).
# See results/figure_data/README.md.
set -euo pipefail

cd "$(dirname "$0")"

# Every figure script does `from results.figures.plotting_style import *`, which
# resolves as a namespace package only with the repo root on the import path.
REPO_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT_DIR="./plotting_scripts"
APP_SCRIPTS="mean_coal_rates.py"   # empirical figures — see generate_app_figs.sh

mkdir -p sim_figs sim_figs_png

for script in "$SCRIPT_DIR"/*.py; do
    name=$(basename "$script")
    case " $APP_SCRIPTS " in *" $name "*) continue ;; esac
    echo "Running $name..."
    python "$script"
done

echo "Simulation figures written to sim_figs/"
