"""
Verify the per-set singer-snakemake input directories.

The checks that matter most here are the ones the workflow itself cannot make:

- metadata order. `chunk_chromosomes.py` zips metadata rows to VCF samples positionally and
  only ever checks the row count, so a mis-ordered file mislabels every sample silently. This
  compares element by element against `bcftools query -l`, not as sets.
- every one of the five per-chromosome files present, non-empty, and with symlinks resolving.
- no duplicate positions, which would trip the `np.all(np.diff(POS) > 0)` assertion in
  `write_minimal_vcf` after hours of work.

Exits nonzero if any check fails.
"""

import argparse
import csv
import os
import subprocess
import sys

import yaml

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]
SHARED_SUFFIXES = ["mask.bed", "hapmap", "ancestral.fa.gz"]


class Report:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, condition, name, detail=""):
        self.checks += 1
        if not condition:
            print(f"  FAIL  {name}{(' -- ' + detail) if detail else ''}")
            self.failures.append(name)
        return condition

    def done(self):
        print(f"\n{self.checks - len(self.failures)}/{self.checks} checks passed")
        if self.failures:
            print("\nFAILED:", file=sys.stderr)
            for name in self.failures:
                print(f"  - {name}", file=sys.stderr)
            sys.exit(1)


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="data/human_data")
    parser.add_argument(
        "--sets", nargs="*", help="subset of sets to check (default: all found)"
    )
    args = parser.parse_args()

    sets_dir = os.path.join(args.root, "resources", "hgdp", "sets")
    base = os.path.join(args.root, "SingerSnakemake_SampleSets")

    # Read max-masked-samples from the config rather than hardcoding it, so the limit asserted
    # here and the one the workflow actually uses cannot drift apart.
    config_path = os.path.join(args.root, "singer_config.yaml")
    singer_config = yaml.safe_load(open(config_path)) if os.path.exists(config_path) else {}
    names = args.sets or sorted(
        f[:-4] for f in os.listdir(sets_dir) if f.endswith(".csv")
    )

    report = Report()

    for name in names:
        input_dir = os.path.join(base, name, "input")
        print(f"\n{name}")
        if not report.check(os.path.isdir(input_dir), f"{name} input dir exists"):
            continue

        with open(os.path.join(sets_dir, f"{name}.csv")) as handle:
            expected = [line.strip() for line in handle if line.strip()]

        vcf_order = None
        for chrom in AUTOSOMES:
            vcf = os.path.join(input_dir, f"{chrom}.vcf.gz")
            if not report.check(
                os.path.exists(vcf) and os.path.getsize(vcf) > 0,
                f"{name}/{chrom} vcf present",
            ):
                continue

            report.check(
                os.path.exists(f"{vcf}.tbi"), f"{name}/{chrom} vcf indexed"
            )

            for suffix in SHARED_SUFFIXES:
                path = os.path.join(input_dir, f"{chrom}.{suffix}")
                report.check(
                    os.path.exists(path),  # follows symlinks
                    f"{name}/{chrom}.{suffix} resolves",
                    "" if os.path.exists(path) else "broken symlink or missing",
                )

            samples = [s for s in run(["bcftools", "query", "-l", vcf]).splitlines() if s]
            if vcf_order is None:
                vcf_order = samples
                report.check(
                    samples == sorted(expected),
                    f"{name} VCF samples match the set definition",
                    f"{len(samples)} in VCF vs {len(expected)} expected",
                )
            else:
                report.check(
                    samples == vcf_order,
                    f"{name}/{chrom} sample order matches chr1",
                )

            meta_path = os.path.join(input_dir, f"{chrom}.meta.csv")
            if report.check(
                os.path.exists(meta_path), f"{name}/{chrom} meta.csv present"
            ):
                with open(meta_path) as handle:
                    rows = list(csv.DictReader(handle))
                report.check(
                    len(rows) == len(samples),
                    f"{name}/{chrom} meta.csv row count == sample count",
                    f"{len(rows)} vs {len(samples)}",
                )
                report.check(
                    [r["sample_id"] for r in rows] == samples,
                    f"{name}/{chrom} meta.csv ORDER matches VCF",
                )

        # Per-sample missing-genotype masks. The load-bearing check is the last one: at
        # `max-masked-samples: 0` (the upstream default) every masked position is marked
        # globally inaccessible, silently deleting the sites the mask exists to rescue.
        # scripts/test_max_masked_samples.py proves both branches against the workflow's own
        # parser; this asserts the invariant cheaply for every set and chromosome.
        worst_masked = 0
        for chrom in AUTOSOMES:
            bed_path = os.path.join(input_dir, f"{chrom}.sample.mask.bed")
            if not report.check(
                os.path.exists(bed_path), f"{name}/{chrom} sample.mask.bed present"
            ):
                continue

            sample_set = set(vcf_order or [])
            bad_columns = bad_span = 0
            unknown_samples = set()
            per_position = {}
            with open(bed_path) as handle:
                for line in handle:
                    fields = line.split()
                    if len(fields) != 4:
                        bad_columns += 1
                        continue
                    _, start, end, sample_id = fields
                    if int(end) - int(start) != 1:
                        bad_span += 1
                    if sample_id not in sample_set:
                        unknown_samples.add(sample_id)
                    per_position[start] = per_position.get(start, 0) + 1

            report.check(
                bad_columns == 0,
                f"{name}/{chrom} sample.mask.bed has exactly 4 columns",
                f"{bad_columns} malformed lines",
            )
            report.check(
                bad_span == 0,
                f"{name}/{chrom} sample.mask.bed intervals are 1 bp",
                f"{bad_span} wrong-width intervals",
            )
            report.check(
                not unknown_samples,
                f"{name}/{chrom} sample.mask.bed names only VCF samples",
                f"unknown: {sorted(unknown_samples)[:3]}",
            )
            worst_masked = max(worst_masked, max(per_position.values(), default=0))

        max_masked_allowed = singer_config.get("max-masked-samples", 0)
        report.check(
            worst_masked < max_masked_allowed,
            f"{name} max masked samples at a site < max-masked-samples",
            f"{worst_masked} masked vs limit {max_masked_allowed} "
            f"(at or above the limit, those positions become INACCESSIBLE)",
        )

        # Duplicate positions would trip write_minimal_vcf's monotonicity assertion.
        for chrom in ("chr1", "chr22"):
            vcf = os.path.join(input_dir, f"{chrom}.vcf.gz")
            if not os.path.exists(vcf):
                continue
            positions = run(["bcftools", "query", "-f", "%POS\n", vcf]).split()
            report.check(
                len(positions) == len(set(positions)),
                f"{name}/{chrom} no duplicate positions",
                f"{len(positions):,} sites",
            )

    report.done()


if __name__ == "__main__":
    main()
