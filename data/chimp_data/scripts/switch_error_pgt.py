#!/usr/bin/env python3
"""
Measure statistical-phasing switch error against GATK read-backed phase blocks.

There are no trios in this panel, so the `PGT`/`PID` fields written by GATK
HaplotypeCaller are the only independent phase information available. Within one `PID`
block the `PGT` values are phased against each other by the reads themselves, so any
two heterozygous sites sharing a `PID` give one testable orientation: either the
statistical phasing agrees with the read-backed orientation or it is switched.

The rate reported is the *adjacent-pair switch error rate*: over consecutive het pairs
inside a shared block, the fraction where the two callers disagree about orientation.
Pairs are only ever compared within a block, never across blocks, because `PID` values
carry no phase relationship to one another.

Reads two VCFs covering the same region:
  --raw     the unphased callset carrying PGT/PID (the truth track)
  --phased  the statistically phased VCF to be judged

Both must contain the same samples. Sites are matched on (CHROM, POS, REF, ALT), so a
polarisation step that swaps REF/ALT must not be applied to either input before this
runs.

Writes a per-subspecies TSV and prints a summary.
"""

import argparse
import csv
import subprocess
import sys
from collections import defaultdict


def read_pgt_blocks(vcf, region, samples):
    """
    Return {sample_index: {pid: {(pos, ref, alt): pgt}}} from the raw callset.

    Only heterozygous read-backed calls are kept: PGT is by construction heterozygous,
    but a missing PGT or PID makes the site unusable, and homozygous sites carry no
    orientation to compare.
    """
    # PGT is itself a phased genotype containing a '|' (e.g. "0|1"), so the two fields are
    # separated by a character that cannot occur inside either.
    fmt = "%CHROM\t%POS\t%REF\t%ALT[\t%PGT/%PID]\n"
    cmd = ["bcftools", "query", "-f", fmt]
    if region:
        cmd += ["-r", region]
    cmd.append(vcf)

    blocks = defaultdict(lambda: defaultdict(dict))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2], parts[3]
        for i, field in enumerate(parts[4:]):
            if "/" not in field:
                continue
            pgt, pid = field.split("/", 1)
            # A site is usable only when both the orientation and its block id are set.
            # PGT must be phased ("0|1"); an unphased or absent value carries no orientation.
            if pid == "." or pgt == "." or "|" not in pgt:
                continue
            blocks[i][pid][(pos, ref, alt)] = pgt
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"bcftools query failed on {vcf}")
    return blocks


def read_phased_calls(vcf, region):
    """Return {(pos, ref, alt): [gt_per_sample]} from the statistically phased VCF."""
    fmt = "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n"
    cmd = ["bcftools", "query", "-f", fmt]
    if region:
        cmd += ["-r", region]
    cmd.append(vcf)

    calls = {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        pos, ref, alt = int(parts[1]), parts[2], parts[3]
        calls[(pos, ref, alt)] = parts[4:]
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"bcftools query failed on {vcf}")
    return calls


def orientation(gt):
    """
    Reduce a phased heterozygous genotype to a binary orientation, or None.

    '0|1' -> 0 and '1|0' -> 1. Anything unphased, homozygous or missing has no
    orientation and cannot contribute to a comparison.
    """
    if "|" not in gt:
        return None
    a, b = gt.split("|", 1)
    if a == b or a not in ("0", "1") or b not in ("0", "1"):
        return None
    return 0 if (a, b) == ("0", "1") else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="Unphased VCF carrying PGT/PID.")
    ap.add_argument("--phased", required=True, help="Statistically phased VCF to judge.")
    ap.add_argument("--subspecies", required=True,
                    help="TSV with sample_id and subspecies columns.")
    ap.add_argument("--region", default=None, help="bcftools region string.")
    ap.add_argument("--out", required=True, help="Per-subspecies TSV to write.")
    args = ap.parse_args()

    # Sample order must agree between the two VCFs, since comparisons are by column index.
    raw_samples = subprocess.run(["bcftools", "query", "-l", args.raw],
                                 capture_output=True, text=True, check=True).stdout.split()
    phased_samples = subprocess.run(["bcftools", "query", "-l", args.phased],
                                    capture_output=True, text=True, check=True).stdout.split()
    if raw_samples != phased_samples:
        raise ValueError(
            "Sample order differs between the raw and phased VCFs; comparisons are made "
            "by column index, so they must match element by element."
        )

    subspecies = {}
    with open(args.subspecies) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            subspecies[row["sample_id"]] = row["subspecies"]

    blocks = read_pgt_blocks(args.raw, args.region, raw_samples)
    phased = read_phased_calls(args.phased, args.region)

    # Per sample, walk each read-backed block in position order and compare the
    # orientation of every consecutive het pair against the statistical phasing.
    per_sample = {}
    for idx, pids in blocks.items():
        name = raw_samples[idx]
        tested = switched = 0
        for pid, sites in pids.items():
            usable = []
            for key in sorted(sites):
                gt = phased.get(key)
                if gt is None:
                    continue
                stat = orientation(gt[idx])
                read = orientation(sites[key])
                if stat is None or read is None:
                    continue
                usable.append((stat, read))
            # Each consecutive pair is one independent orientation test.
            for (s1, r1), (s2, r2) in zip(usable, usable[1:]):
                tested += 1
                if (s1 == s2) != (r1 == r2):
                    switched += 1
        if tested:
            per_sample[name] = (tested, switched)

    if not per_sample:
        raise RuntimeError(
            "No comparable het pairs found. Check that the two VCFs cover the same "
            "region and that the raw VCF actually carries PGT/PID."
        )

    # Aggregate to subspecies, which is the level the phasing quality question is asked at.
    agg = defaultdict(lambda: [0, 0, 0])
    for name, (tested, switched) in per_sample.items():
        grp = subspecies.get(name, "UNKNOWN")
        agg[grp][0] += tested
        agg[grp][1] += switched
        agg[grp][2] += 1

    with open(args.out, "w") as fh:
        fh.write("subspecies\tn_samples\tpairs_tested\tswitches\tswitch_error_rate\n")
        for grp in sorted(agg):
            tested, switched, n = agg[grp]
            fh.write(f"{grp}\t{n}\t{tested}\t{switched}\t{switched / tested:.6f}\n")
        tot_t = sum(v[0] for v in agg.values())
        tot_s = sum(v[1] for v in agg.values())
        fh.write(f"ALL\t{len(per_sample)}\t{tot_t}\t{tot_s}\t{tot_s / tot_t:.6f}\n")

    print(f"{'subspecies':<24}{'n':>4}{'pairs':>12}{'switches':>10}{'rate':>10}")
    for grp in sorted(agg):
        tested, switched, n = agg[grp]
        print(f"{grp:<24}{n:>4}{tested:>12,}{switched:>10,}{switched / tested:>10.4f}")
    tot_t = sum(v[0] for v in agg.values())
    tot_s = sum(v[1] for v in agg.values())
    print(f"{'ALL':<24}{len(per_sample):>4}{tot_t:>12,}{tot_s:>10,}{tot_s / tot_t:>10.4f}")


if __name__ == "__main__":
    sys.exit(main())
