"""
Verify the shared per-chromosome resources before any SINGER run is launched.

These are invariants, not one-off checks. Each corresponds to a failure that is either silent
(mask polarity, metadata order) or surfaces only hours into a run (the exact-length assert), so
they are worth re-running whenever the resources are rebuilt.

Exits nonzero if any check fails.
"""

import argparse
import gzip
import os
import sys

import numpy as np
from msprime import RateMap

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

# Measured accessible fraction is 48%-82% (mean ~74%); chr22 is lowest because its entire
# acrocentric p-arm (0-10.66 Mb) is unmappable satellite. See make_masks.py for the full
# rationale. This band tests for INVERTED polarity, which would put most chromosomes near 26%.
MIN_ACCESSIBLE = 0.40
MAX_ACCESSIBLE = 0.95

# Sex-averaged deCODE totals are ~0.5-2.5 cM/Mb genome-wide; small chromosomes recombine faster.
MIN_CM_PER_MB = 0.5
MAX_CM_PER_MB = 3.0


class Report:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, condition, name, detail=""):
        self.checks += 1
        if condition:
            print(f"  ok    {name}{(' -- ' + detail) if detail else ''}")
        else:
            print(f"  FAIL  {name}{(' -- ' + detail) if detail else ''}")
            self.failures.append(name)

    def done(self):
        print(f"\n{self.checks - len(self.failures)}/{self.checks} checks passed")
        if self.failures:
            print("\nFAILED:", file=sys.stderr)
            for name in self.failures:
                print(f"  - {name}", file=sys.stderr)
            sys.exit(1)


def fasta_length(path):
    """Measure a gzipped FASTA the way read_single_fasta does, and count records."""
    length = 0
    headers = 0
    first_line_is_header = False
    with gzip.open(path, "rt") as handle:
        for i, line in enumerate(handle):
            if line.startswith(">"):
                headers += 1
                if i == 0:
                    first_line_is_header = True
            else:
                length += len(line.strip())
    return length, headers, first_line_is_header


def read_chrom_sizes(path):
    sizes = {}
    with open(path) as handle:
        for line in handle:
            name, value = line.split()[:2]
            sizes[name] = int(value)
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", required=True, help="resources/ directory")
    args = parser.parse_args()

    grch38 = os.path.join(args.resources, "grch38")
    chrom_sizes_path = os.path.join(grch38, "chrom.sizes")
    report = Report()

    print("chrom.sizes")
    report.check(os.path.exists(chrom_sizes_path), "chrom.sizes exists")
    if not os.path.exists(chrom_sizes_path):
        report.done()
    sizes = read_chrom_sizes(chrom_sizes_path)
    report.check(
        all(c in sizes for c in AUTOSOMES),
        "all 22 autosomes present",
        f"{len(sizes)} entries",
    )

    print("\nancestral FASTA (single record, spans chromosome)")
    fasta_lengths = {}
    for chrom in AUTOSOMES:
        path = os.path.join(grch38, "ancestral", f"{chrom}.ancestral.fa.gz")
        if not os.path.exists(path):
            report.check(False, f"{chrom} ancestral FASTA exists")
            continue
        length, headers, header_first = fasta_length(path)
        fasta_lengths[chrom] = length
        report.check(
            headers == 1 and header_first,
            f"{chrom} single FASTA record",
            f"{headers} header(s)",
        )
        report.check(
            length == sizes.get(chrom),
            f"{chrom} FASTA length == chrom.sizes",
            f"{length:,} vs {sizes.get(chrom):,}",
        )

    print("\nhapmap (sequence_length must match the FASTA EXACTLY)")
    for chrom in AUTOSOMES:
        path = os.path.join(grch38, "hapmap", f"{chrom}.hapmap")
        if not os.path.exists(path):
            report.check(False, f"{chrom} hapmap exists")
            continue
        rate_map = RateMap.read_hapmap(path)
        span = int(rate_map.sequence_length)
        report.check(
            span == sizes.get(chrom),
            f"{chrom} hapmap sequence_length == chrom.sizes",
            f"{span:,} vs {sizes.get(chrom):,}",
        )
        if chrom in fasta_lengths:
            report.check(
                span == fasta_lengths[chrom],
                f"{chrom} hapmap == ancestral FASTA length",
                f"{span:,} vs {fasta_lengths[chrom]:,}",
            )

        cumulative = rate_map.get_cumulative_mass(rate_map.sequence_length) * 100
        cm_per_mb = cumulative / (span / 1e6)
        report.check(
            MIN_CM_PER_MB <= cm_per_mb <= MAX_CM_PER_MB,
            f"{chrom} mean recombination rate plausible",
            f"{cumulative:.1f} cM total, {cm_per_mb:.2f} cM/Mb",
        )

        positions, cm = np.loadtxt(
            path, skiprows=1, usecols=[1, 3], unpack=True, dtype=float
        )
        report.check(
            bool(np.all(np.diff(cm) >= 0)), f"{chrom} cumulative cM non-decreasing"
        )
        report.check(
            bool(np.all(np.diff(positions) > 0)), f"{chrom} positions strictly increasing"
        )

    print("\nmask (must list INACCESSIBLE intervals)")
    for chrom in AUTOSOMES:
        path = os.path.join(grch38, "mask", f"{chrom}.mask.bed")
        if not os.path.exists(path):
            report.check(False, f"{chrom} mask exists")
            continue
        intervals = np.loadtxt(path, usecols=[1, 2], ndmin=2).astype(np.int64)
        length = sizes.get(chrom, 0)
        inaccessible = int(np.diff(intervals, axis=1).sum()) if intervals.size else 0
        accessible_fraction = (length - inaccessible) / length
        report.check(
            MIN_ACCESSIBLE <= accessible_fraction <= MAX_ACCESSIBLE,
            f"{chrom} accessible fraction in range (polarity)",
            f"{accessible_fraction:.2%} accessible",
        )
        report.check(
            intervals.size == 0 or int(intervals.max()) <= length,
            f"{chrom} mask within chromosome bounds",
            f"max {int(intervals.max()) if intervals.size else 0:,} <= {length:,}",
        )

    report.done()


if __name__ == "__main__":
    main()
