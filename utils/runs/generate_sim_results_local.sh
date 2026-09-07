#!/bin/bash
# Run every sim CoaSTT run sequentially on this machine.
# Each run script under local_runs/sim_runs/ is self-anchoring, so this can be
# invoked from anywhere. Extra arguments are forwarded to snakemake
# (e.g. -n for a dry run).
set -euo pipefail

cd "$(dirname "$0")"
RUN_DIR="local_runs/sim_runs"

shopt -s nullglob
scripts=("$RUN_DIR"/*.sh)
if [[ ${#scripts[@]} -eq 0 ]]; then
    echo "No run scripts in $RUN_DIR — nothing to do."
    exit 0
fi

echo "Running ${#scripts[@]} sim run(s) sequentially."
for script in "${scripts[@]}"; do
    echo "=== $(basename "$script") ==="
    bash "$script" "$@"
done
echo "All sim runs complete."
