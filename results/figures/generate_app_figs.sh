#!/bin/bash
# Regenerate every application (empirical: human / chimp) figure into app_figs/.
# Each script is self-anchoring (it chdir's to its own directory), so this can be
# run from anywhere. PNG intermediates are written to app_figs_png/.
set -euo pipefail

cd "$(dirname "$0")"
SCRIPT_DIR="./plotting_scripts"
APP_SCRIPTS="mean_coal_rates.py"

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

echo "Application figures written to app_figs/"
