#!/usr/bin/env python3
"""
Assert the singer-snakemake input invariants for one or more chromosomes.

Every check here covers a failure that is otherwise silent or very late: the workflow
asserts exact equality between the hapmap length, the ancestral FASTA length and the
chromosome length, and it would otherwise fail hours into a run. Exits non-zero on the
first failure, so this is usable as a pipeline gate.

Run from data/chimp_data:
    python3 scripts/verify_inputs.py chr21
    python3 scripts/verify_inputs.py            # every chromosome in chrom.sizes
"""

import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
from msprime import RateMap

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "resources" / "panTro6"

# Accessible-fraction band. The check exists to catch INVERTED POLARITY -- a mask storing
# accessible rather than inaccessible intervals -- not to enforce a target retention.
#
# The band is deliberately wider and higher than the human prep's. That mask combines
# mappability, read depth and excess heterozygosity and retains ~74%; this one can only ever
# have the mappability component, because the callset is variant-only and depth cannot be
# assessed where nothing was called. Measured retention on panTro6 chr21 is 94.4% at k=36,
# e=0 -- high because a PacBio assembly is almost entirely uniquely mappable, not because
# the mask is broken. An inverted mask would land near 5%, far below the floor.
ACCESSIBLE_MIN = 0.50
ACCESSIBLE_MAX = 0.99

failures = []
checks = 0


def check(ok, label, detail=""):
    """Record one assertion. Never raises, so a run reports every failure rather than one."""
    global checks
    checks += 1
    if ok:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f" -- {detail}" if detail else ""))
        failures.append(label)


def chrom_sizes():
    sizes = {}
    for line in open(RES / "chrom.sizes"):
        name, length = line.split()
        sizes[name] = int(length)
    return sizes


def verify_hapmap(chrom, length):
    path = RES / "hapmap" / f"{chrom}.hapmap"
    if not path.exists():
        check(False, f"{chrom}: hapmap exists", str(path))
        return
    hm = RateMap.read_hapmap(str(path))
    # Exact equality, not sufficiency: chunk_chromosomes.py compares with ==.
    check(int(hm.sequence_length) == length,
          f"{chrom}: hapmap sequence_length == chrom.sizes",
          f"{int(hm.sequence_length)} != {length}")
    check(bool(np.all(np.diff(hm.position) > 0)),
          f"{chrom}: hapmap positions strictly increasing")
    finite = hm.rate[np.isfinite(hm.rate)]
    check(bool(np.all(finite >= 0)), f"{chrom}: no negative recombination rates")

    cm = np.array([float(line.split("\t")[3]) for line in open(path).readlines()[1:]])
    check(bool(np.all(np.diff(cm) >= -1e-9)),
          f"{chrom}: cumulative cM non-decreasing")
    mean_rate = cm[-1] / (length / 1e6)
    check(0.3 < mean_rate < 4.0,
          f"{chrom}: mean cM/Mb within 0.3-4.0", f"{mean_rate:.3f}")


def verify_mask(chrom, length):
    path = RES / "mask" / f"{chrom}.mask.bed"
    if not path.exists():
        check(False, f"{chrom}: mask exists", str(path))
        return
    bed = np.loadtxt(path, usecols=[1, 2], ndmin=2).astype(np.int64)
    if bed.size == 0:
        check(False, f"{chrom}: mask is non-empty")
        return
    check(int(bed.max()) <= length, f"{chrom}: mask intervals within chromosome bounds")
    check(bool(np.all(bed[:, 1] > bed[:, 0])), f"{chrom}: mask intervals well-formed")
    inaccessible = int(np.sum(bed[:, 1] - bed[:, 0]))
    frac = 1 - inaccessible / length
    check(ACCESSIBLE_MIN <= frac <= ACCESSIBLE_MAX,
          f"{chrom}: accessible fraction in [{ACCESSIBLE_MIN}, {ACCESSIBLE_MAX}] "
          f"-- mask MUST contain INACCESSIBLE intervals, not accessible ones",
          f"{frac:.4f}")
    print(f"        accessible fraction: {frac:.4f}")


def verify_ancestral(chrom, length):
    path = RES / "ancestral" / f"{chrom}.ancestral.fa.gz"
    if not path.exists():
        check(False, f"{chrom}: ancestral FASTA exists", str(path))
        return
    headers = 0
    n = 0
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                headers += 1
            else:
                n += len(line.strip())
    # read_single_fasta asserts exactly one record.
    check(headers == 1, f"{chrom}: ancestral FASTA is a single record", f"{headers} headers")
    check(n == length, f"{chrom}: ancestral FASTA length == chrom.sizes", f"{n} != {length}")


def verify_vcf(chrom):
    path = ROOT / "SingerSnakemake_Input" / f"{chrom}.vcf.gz"
    if not path.exists():
        return  # inputs not assembled yet; the per-resource checks above still apply
    out = subprocess.run(["bcftools", "query", "-f", "%POS\n", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    pos = np.array(out, dtype=np.int64)
    check(pos.size == np.unique(pos).size, f"{chrom}: no duplicate VCF positions")

    # Every record must be a biallelic SNP. Non-biallelic records are removed in step 1 and
    # never reintroduced, so anything here means the filter or the assembly regressed.
    n_all = int(subprocess.run(["bcftools", "index", "-n", str(path)],
                               capture_output=True, text=True, check=True).stdout.strip())
    n_bi = int(subprocess.run(
        ["bash", "-c", f"bcftools view -H -m2 -M2 -v snps {path} | wc -l"],
        capture_output=True, text=True, check=True).stdout.strip())
    check(n_all == n_bi, f"{chrom}: every record is a biallelic SNP",
          f"{n_all - n_bi:,} of {n_all:,} records are not")

    # SINGER consumes HAPLOTYPES. An unphased input is the one error here that is both fatal
    # to the analysis and invisible to every structural check: the file has the right length,
    # positions and samples, and is simply wrong.
    #
    # Not hypothetical -- it shipped. chr21 once passed 13/13 while carrying 91.79% unphased
    # genotypes, because the phased VCF existed in intermediates/phased/ but was never
    # promoted into the input directory.
    gts = subprocess.run(["bcftools", "query", "-f", "[%GT\n]", str(path)],
                         capture_output=True, text=True, check=True).stdout.split()
    sample = gts[:500_000]
    unphased = sum(1 for g in sample if "/" in g)
    check(unphased == 0,
          f"{chrom}: all genotypes phased (must be the SHAPEIT5 output, not the "
          f"pre-phase subset)",
          f"{unphased:,} of {len(sample):,} sampled genotypes contain '/'")

    meta = ROOT / "SingerSnakemake_Input" / f"{chrom}.meta.csv"
    if meta.exists():
        samples = subprocess.run(["bcftools", "query", "-l", str(path)],
                                 capture_output=True, text=True, check=True).stdout.split()
        rows = [ln.split(",")[0] for ln in meta.read_text().splitlines()[1:]]
        # Zipped by index with no ID matching, so order must match element by element.
        check(rows == samples, f"{chrom}: meta.csv order matches bcftools query -l")
        bonobos = {"SAMN01920505", "SAMN01920507", "SAMN01920508", "SAMN01920509",
                   "SAMN01920510", "SAMN01920512", "SAMN01920515"}
        check(not (set(samples) & bonobos), f"{chrom}: no bonobo samples present")


def main():
    sizes = chrom_sizes()
    targets = sys.argv[1:] or sorted(sizes)
    for chrom in targets:
        if chrom not in sizes:
            print(f"unknown chromosome {chrom}", file=sys.stderr)
            return 2
        print(f"\n=== {chrom} ({sizes[chrom]:,} bp) ===")
        verify_hapmap(chrom, sizes[chrom])
        verify_mask(chrom, sizes[chrom])
        verify_ancestral(chrom, sizes[chrom])
        verify_vcf(chrom)

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED:", "; ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
