#!/usr/bin/env python3
"""Drive the ARG-inference rates comparison through the existing snakemake pipeline.

One demography, one stage per invocation. The stage ``sim`` simulates the tree
sequences and computes both the analytic true rates and the CoaSTT estimate on
those true trees; each method stage (``tsinfer``, ``relate``, ``singer``) infers
trees from the same simulated contigs and computes the estimate on them. Nothing
here re-implements simulation, inference or rate estimation: the script only
derives a config from the demography's tracked config and hands it to snakemake.

Layout produced under --root:

    <root>/<demography>/configs/<stage>.yaml   the derived config (provenance)
    <root>/<demography>/sim/                   simulated trees, rates_true, rates
    <root>/<demography>/<method>/              inferred trees, rates

The method directories sit beside ``sim`` because the snakefile derives
``sim_trees_dir`` by swapping the method name in ``output_path`` for "sim".

Usage:
    python misc_utils/arg_rates_comparison.py --demography split --stage singer
"""
from __future__ import annotations

import argparse
import os
import subprocess

import yaml

# All pipeline paths are relative to utils/, so snakemake is always run from there.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.dirname(SCRIPT_DIR)

DEMOGRAPHIES = ("split", "sawtooth")
METHODS = ("tsinfer", "relate", "singer")
STAGES = ("sim",) + METHODS

DEFAULT_ROOT = "../results/outputs/misc_analyses/arg_rates_comparison"

# Fixed for the comparison, overriding whatever the tracked config carries.
CONTIG_LENGTH = 5000000
N_SIM_PER_POP = 20
NUM_CONTIGS = 500
SINGER_BURNIN = 2500
# 5 samples 500 iterations apart; infer_trees keeps only the last
SINGER_THINNING = 500
# min_window_size 40 lets the rates start at 20 generations. Sawtooth max_window_size
# 6000 gave the lowest error of the sim-tree mean rate against the truth (100-contig
# runs); the inferred-tree means were flat (within 0.1%) over 1000-16000
RATE_OVERRIDES = {
    "split": {"min_window_size": 40, "rates_min_time": 20},
    "sawtooth": {"min_window_size": 40, "rates_min_time": 20, "max_window_size": 6000},
}


def build_config(demography, stage, root, num_contigs, num_threads):
    """Derive the stage's config from the demography's tracked sim config."""
    base_path = os.path.join(UTILS_DIR, "..", "data", "configs", "sim_configs",
                             f"{demography}.yaml")
    with open(base_path) as f:
        config = yaml.safe_load(f)

    config["num_contigs"] = num_contigs
    config["contig_length"] = CONTIG_LENGTH
    config["n_sim_per_pop"] = N_SIM_PER_POP
    config["num_threads"] = num_threads
    config.update(RATE_OVERRIDES.get(demography, {}))
    config["burnin"] = SINGER_BURNIN
    config["thinning_interval"] = SINGER_THINNING
    # sim writes to <root>/<demo>/sim; a method writes to the sibling <root>/<demo>/<method>,
    # which is where the snakefile looks for the simulated trees (sim_trees_dir).
    config["output_path"] = f"{root}/{demography}/{'sim' if stage == 'sim' else stage}"
    if stage != "sim":
        config["infer_method"] = stage
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demography", required=True, choices=DEMOGRAPHIES)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--num_contigs", type=int, default=NUM_CONTIGS)
    p.add_argument("--num_threads", type=int, default=24)
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="run root, relative to utils/ (default: %(default)s)")
    p.add_argument("--dry_run", action="store_true", help="pass -n to snakemake")
    args = p.parse_args()

    config = build_config(args.demography, args.stage, args.root,
                          args.num_contigs, args.num_threads)

    # The derived config is written next to the outputs it produces, so a run can
    # be reproduced from the directory it landed in.
    config_dir = os.path.join(UTILS_DIR, args.root, args.demography, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"{args.stage}.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # sim also produces the analytic true rates and the estimate on the true trees;
    # a method stage only needs compute_rates, which pulls infer_trees -> compute_pdf.
    targets = (["sim", "compute_rates_true", "compute_rates"] if args.stage == "sim"
               else ["compute_rates"])

    cmd = ["snakemake", *targets,
           "--configfile", os.path.relpath(config_path, UTILS_DIR),
           "--cores", str(args.num_threads),
           "--rerun-incomplete", "--nolock"]
    if args.dry_run:
        cmd.append("-n")

    print(f"[arg_rates_comparison] config: {config_path}")
    print(f"[arg_rates_comparison] cwd: {UTILS_DIR}")
    print(f"[arg_rates_comparison] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=UTILS_DIR, check=True)


if __name__ == "__main__":
    main()
