#!/bin/bash
# Regenerate every simulation figure into sim_figs/.
# Each script is self-anchoring (it chdir's to its own directory), so this can be
# run from anywhere. PNG intermediates are written to sim_figs_png/.
set -euo pipefail

cd "$(dirname "$0")"
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
