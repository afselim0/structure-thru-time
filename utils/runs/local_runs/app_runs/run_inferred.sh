#!/bin/bash
# Shared body for the app_runs scripts: run CoaSTT on inferred trees.
#
# Usage (sourced, not executed, from a per-run script):
#   run_inferred <config_basename> <infer_method> [extra --config KEY=VALUE ...]
#
# Each demography's config writes its simulated trees to an output_path ending
# in /sim. This redirects output_path to a sibling directory named after the
# inference method, so both tree sources live side by side under one run dir:
#   ../results/outputs/simulations/<demography>/sim/trees      <- simulated
#   ../results/outputs/simulations/<demography>/<method>/trees <- inferred
# The snakefile's vcf_dir defaults to tree_output_path with infer_method
# swapped back to "sim", so infer_trees picks up the simulated VCFs with no
# explicit path. The matching sim_runs script must be run first.
run_inferred() {
    local config_name="$1"; shift
    local infer_method="$1"; shift

    local configfile="../data/configs/sim_configs/${config_name}.yaml"
    if [ ! -f "$configfile" ]; then
        echo "run_inferred: config not found: $configfile" >&2
        return 1
    fi

    # Replace the trailing "sim" component of the config's output_path with the
    # inference method, keeping the rest of the run directory intact.
    local sim_output_dir
    sim_output_dir="$(grep -E '^output_path:' "$configfile" | head -n1 | sed -E 's/^output_path:[[:space:]]*//')"
    if [ -z "$sim_output_dir" ]; then
        echo "run_inferred: no output_path in $configfile" >&2
        return 1
    fi
    if [ "${sim_output_dir%/sim}" = "$sim_output_dir" ]; then
        echo "run_inferred: output_path '$sim_output_dir' does not end in /sim" >&2
        return 1
    fi
    local output_dir="${sim_output_dir%/sim}/${infer_method}"

    snakemake STT_infer \
        --configfile "$configfile" \
        --config output_path="$output_dir" infer_method="$infer_method" "$@" \
        --cores all --rerun-incomplete "${SNAKEMAKE_ARGS[@]}"
}
