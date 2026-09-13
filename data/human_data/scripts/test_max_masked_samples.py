"""
Regression test for the `max-masked-samples` trap.

This is deliberately a permanent test rather than a one-off check. `max-masked-samples`
defaults to 0, and at that value singer-snakemake marks every position covered by a per-sample
mask as globally INACCESSIBLE -- silently deleting exactly the sites the mask exists to rescue.
Nothing in the upstream config documents the key, and the failure is invisible: the run
succeeds, it just quietly throws away the data.

The only thing standing between that default and a ruined run is one line of our YAML, so this
test asserts BOTH branches against singer-snakemake's own code:

  * at our configured value -> no interval is marked inaccessible
  * at 0                    -> intervals ARE marked inaccessible (the trap is real)

Run:
    python scripts/test_max_masked_samples.py --root . [--set OCEANIA] [--chrom chr22]
"""

import argparse
import os
import sys

import numpy as np
import yaml

SINGER_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "singer-snakemake", "workflow", "scripts",
)


def load_singer_helpers():
    """Import the workflow's own parser rather than reimplementing it, so this test tracks
    upstream behaviour instead of our understanding of it."""
    if SINGER_SCRIPTS not in sys.path:
        sys.path.insert(0, SINGER_SCRIPTS)
    try:
        from utils import parse_sample_bedmask, interval_coverage
    except ImportError as exc:
        raise SystemExit(
            f"Cannot import singer-snakemake helpers from {SINGER_SCRIPTS}: {exc}\n"
            f"Is data/singer-snakemake/ cloned?"
        )
    return parse_sample_bedmask, interval_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="data/human_data")
    parser.add_argument("--set", dest="set_name", default="OCEANIA")
    parser.add_argument("--chrom", default="chr22")
    args = parser.parse_args()

    parse_sample_bedmask, interval_coverage = load_singer_helpers()

    config_path = os.path.join(args.root, "drivers", "singer_config.yaml")
    config = yaml.safe_load(open(config_path))
    configured = config.get("max-masked-samples", 0)

    input_dir = os.path.join(
        args.root, "SingerSnakemake_SampleSets", args.set_name
    )
    bed = os.path.join(input_dir, f"{args.chrom}.sample.mask.bed")
    vcf = os.path.join(input_dir, f"{args.chrom}.vcf.gz")
    for path in (bed, vcf):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path} -- run subset.sh first")

    import subprocess
    samples = [
        s for s in subprocess.run(
            ["bcftools", "query", "-l", vcf], check=True, capture_output=True, text=True
        ).stdout.splitlines() if s
    ]
    sample_map = {name: i for i, name in enumerate(samples)}

    samplemask = parse_sample_bedmask(bed, sample_map)
    intervals, coverage = interval_coverage(samplemask)
    worst = int(coverage.max()) if coverage.size else 0

    failures = []

    def check(condition, name, detail=""):
        mark = "ok   " if condition else "FAIL "
        print(f"  {mark} {name}{(' -- ' + detail) if detail else ''}")
        if not condition:
            failures.append(name)

    print(f"{args.set_name}/{args.chrom}: {len(samplemask)} masked samples, "
          f"max {worst} samples masked at any one position")

    check(
        len(samplemask) > 0,
        "mask parses to at least one sample",
        f"{len(samplemask)} samples",
    )

    # The configured value must clear the observed maximum.
    check(
        configured > worst,
        f"configured max-masked-samples ({configured}) exceeds observed max ({worst})",
    )

    # Branch 1: at our value, nothing is marked inaccessible.
    inaccessible_ok = intervals[coverage > configured]
    check(
        inaccessible_ok.shape[0] == 0,
        f"at max-masked-samples={configured}, no interval is marked inaccessible",
        f"{inaccessible_ok.shape[0]} intervals",
    )

    # Branch 2: at the default of 0, the trap fires. If this does NOT fire, the test has
    # stopped being meaningful -- either the mask is empty or upstream changed the semantics.
    inaccessible_trap = intervals[coverage > 0]
    trapped_bp = int(np.diff(inaccessible_trap, axis=1).sum()) if inaccessible_trap.size else 0
    check(
        inaccessible_trap.shape[0] > 0,
        "at max-masked-samples=0, intervals ARE marked inaccessible (trap is real)",
        f"{inaccessible_trap.shape[0]} intervals, {trapped_bp} bp would be lost",
    )

    print()
    if failures:
        print("FAILED:", file=sys.stderr)
        for name in failures:
            print(f"  - {name}", file=sys.stderr)
        sys.exit(1)
    print(f"All checks passed. The default of 0 would discard {trapped_bp:,} bp "
          f"on {args.set_name}/{args.chrom} alone.")


if __name__ == "__main__":
    main()
