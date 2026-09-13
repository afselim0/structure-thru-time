#!/usr/bin/env python3
"""
Delete the pre-convergence tree sequences that singer-snakemake writes but nothing uses.

`singer-mcmc-burnin` is applied ONLY to statistics and diagnostic plots; `merge_chunks`
iterates over range(singer-mcmc-samples), so the workflow constructs, dates and STORES every
draw -- all 100 at the current settings, of which only the last 20 are past burn-in. The
other 80 are dead weight the moment they are written: ~240 GB per sample set, on a filesystem
that was at 94% when this was written.

Shared by both preps. Point --config at the config that produced the run, because the burn-in
floor is read from it:

    python scripts/prune_burnin_draws.py --singer-output <dir> --config <singer_config.yaml>
    python scripts/prune_burnin_draws.py --singer-output <dir> --config <cfg> --apply

DRY RUN BY DEFAULT. Without --apply it lists what it would delete and the space it would
free, and removes nothing.

Safety, in order of how much each one matters:

  * Deletes strictly BELOW the burn-in floor. Draws at or above it are what CoaSTT consumes.
  * REFUSES to run unless ts_chunks/ already exists and is non-empty, i.e. extraction has
    already succeeded. Pruning first would delete the inputs of a step that had not yet run,
    and a failed extraction would then be unrecoverable without redoing the inference.
  * Never touches the `.log` files beside the tree sequences: they are small and carry the
    MCMC trace that justifies the burn-in choice in the first place.
"""

import argparse
import os
import sys

import yaml


def burnin_floor(config_path):
    cfg = yaml.safe_load(open(config_path))
    n = int(cfg["singer-mcmc-samples"])
    floor = int(float(cfg["singer-mcmc-burnin"]) * n)
    return n, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singer-output", required=True,
                    help="the singer-snakemake output directory (holds <chrom>/trees/)")
    ap.add_argument("--config", required=True,
                    help="the singer_config.yaml that produced this run")
    ap.add_argument("--ts-chunks", default=None,
                    help="ts_chunks/ to check for; defaults to a sibling of --singer-output")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete; without this it is a dry run")
    args = ap.parse_args()

    n, floor = burnin_floor(args.config)
    print(f"burn-in floor: keeping replicates >= {floor} of {n}")
    if floor <= 0:
        print("floor is 0 -- nothing is inside burn-in, nothing to prune.")
        return 0

    # Extraction must already have happened: these draws are its input.
    ts_chunks = args.ts_chunks or os.path.join(
        os.path.dirname(os.path.abspath(args.singer_output.rstrip("/"))), "ts_chunks")
    if not (os.path.isdir(ts_chunks) and any(
            f.endswith(".trees") for f in os.listdir(ts_chunks))):
        print(f"REFUSING: no extracted tree sequences at {ts_chunks}\n"
              f"Run the extraction step first -- these draws are its input, and deleting "
              f"them before it has succeeded is unrecoverable without redoing the inference.",
              file=sys.stderr)
        return 1

    total_bytes = 0
    total_files = 0
    for chrom in sorted(os.listdir(args.singer_output)):
        trees = os.path.join(args.singer_output, chrom, "trees")
        if not os.path.isdir(trees):
            continue
        doomed = []
        for name in os.listdir(trees):
            if not (name.startswith(f"{chrom}.") and name.endswith(".tsz")):
                continue
            middle = name[len(chrom) + 1:-len(".tsz")]
            if middle.isdigit() and int(middle) < floor:
                doomed.append(name)
        if not doomed:
            continue
        size = sum(os.path.getsize(os.path.join(trees, f)) for f in doomed)
        total_bytes += size
        total_files += len(doomed)
        print(f"  {chrom}: {len(doomed)} draws below {floor}, {size / 1e9:.1f} GB")
        if args.apply:
            for f in doomed:
                os.remove(os.path.join(trees, f))

    verb = "deleted" if args.apply else "would delete"
    print(f"\n{verb} {total_files} tree sequences, {total_bytes / 1e9:.1f} GB")
    if not args.apply and total_files:
        print("Dry run -- nothing removed. Re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
