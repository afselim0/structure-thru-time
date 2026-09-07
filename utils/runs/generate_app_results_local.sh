#!/bin/bash
# Run every app CoaSTT run sequentially on this machine.
# Each run script under local_runs/app_runs/ is self-anchoring, so this can be
# invoked from anywhere. Extra arguments are forwarded to snakemake
# (e.g. -n for a dry run).
set -euo pipefail

cd "$(dirname "$0")"
RUN_DIR="local_runs/app_runs"

shopt -s nullglob
scripts=("$RUN_DIR"/*.sh)
if [[ ${#scripts[@]} -eq 0 ]]; then
    echo "No run scripts in $RUN_DIR — nothing to do."
    exit 0
fi

echo "Running ${#scripts[@]} app run(s) sequentially."
for script in "${scripts[@]}"; do
    echo "=== $(basename "$script") ==="
    bash "$script" "$@"
done
echo "All app runs complete."
