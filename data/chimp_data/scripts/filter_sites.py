#!/usr/bin/env python3
"""
Apply site-quality filters to one chromosome and record what was removed.

Two filter families, both targeting the same dominant failure mode in a non-model assembly
-- collapsed paralogues and mismapping, which masquerade as high-diversity regions and are
exactly the signal CoaSTT reads:

  * GATK hard filters (Best Practices thresholds, human-tuned, NOT re-derived for chimp):
        QD < 2 | FS > 60 | MQ < 40 | SOR > 3 | ReadPosRankSum < -8 | MQRankSum < -12.5
  * Allelic imbalance: pooled alt fraction across heterozygous genotypes outside
        [0.25, 0.75]. A true heterozygote sits near 0.5; a collapsed paralogue does not.
  * Depth: summed FORMAT/DP outside [0.25x, 2x] the genome-wide median.

MUST run BEFORE phasing. SHAPEIT5 emits GT only -- it strips AD, DP and every other FORMAT
field -- so neither the depth nor the allelic-imbalance filter can be computed on a phased
VCF. Measured, not assumed: the phased chr21 carries `##FORMAT=<ID=GT>` and nothing else.

Removed positions are written to <chrom>.filter.txt, which is what scales `adj_mu` in the
workflow. That file is load-bearing: a site removed here but absent from filter.txt inflates
the inferred mutation rate, because the workflow believes the sequence was surveyed and
found monomorphic.

What does NOT belong in filter.txt, and why:
  * Multiallelic sites -- the workflow's own filter_biallelic drops them AND adds them to
    filtered_positions. Recording them here would double-count.
  * Sites in inaccessible regions -- the workflow strips those from filtered_positions
    (`filtered_positions[~bitmask]`), so listing them is a no-op at best.

Run from data/chimp_data:
    python3 scripts/filter_sites.py chr21
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

GATK_EXPR = ("QD<2.0 || FS>60.0 || MQ<40.0 || SOR>3.0 || "
             "ReadPosRankSum<-8.0 || MQRankSum<-12.5")
AI_LOW, AI_HIGH = 0.25, 0.75
# Depth band as multiples of the MEDIAN summed FORMAT/DP. Median not mean: the mean is
# pulled up by exactly the collapsed-paralogue pileups the high cut targets, so a
# mean-based threshold is set too permissively by the contamination it is meant to catch.
DEPTH_LOW_MULT, DEPTH_HIGH_MULT = 0.25, 2.0


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout


def depth_failures(vcf, median_dp):
    """
    Positions whose summed FORMAT/DP falls outside [0.25x, 2x] the genome-wide median.

    Built on FORMAT/DP, never INFO/DP: measured on this callset, INFO/DP runs 8.66x the
    summed per-sample depth because it counts pre-filter reads over all 61 originally-called
    samples. The plan's multiples applied to INFO/DP would reject almost every site.

    The high tail matters most -- collapsed paralogues masquerade as high-diversity regions,
    which is the signal CoaSTT reads. The low tail catches calls resting on too few reads.
    """
    lo, hi = DEPTH_LOW_MULT * median_dp, DEPTH_HIGH_MULT * median_dp
    out = run(f"bcftools query -f '%POS[\t%DP]\n' {vcf}")
    bad = []
    for line in out.splitlines():
        parts = line.split("\t")
        vals = [int(x) for x in parts[1:] if x not in (".", "")]
        if not vals:
            continue            # no depth anywhere: cannot be shown out of band
        total = sum(vals)
        if total < lo or total > hi:
            bad.append(int(parts[0]))
    return np.array(bad, dtype=np.int64), lo, hi


def median_summed_depth(vcf):
    """Median summed FORMAT/DP, sampling every 3rd site to keep the pass cheap."""
    out = run(f"bcftools query -f '%POS[\t%DP]\n' {vcf}")
    tot = []
    for i, line in enumerate(out.splitlines()):
        if i % 3:
            continue
        vals = [int(x) for x in line.split("\t")[1:] if x not in (".", "")]
        if vals:
            tot.append(sum(vals))
    return float(np.median(tot))


def allelic_imbalance_failures(vcf):
    """Positions whose pooled alt fraction over heterozygous calls falls outside the band."""
    out = run(f"bcftools query -f '%POS[\\t%GT:%AD]\\n' {vcf}")
    bad = []
    for line in out.splitlines():
        parts = line.split("\t")
        ref_tot = alt_tot = 0
        for field in parts[1:]:
            if ":" not in field:
                continue
            gt, ad = field.split(":", 1)
            alleles = gt.replace("|", "/").split("/")
            # Only heterozygous calls carry information about allelic balance.
            if len(alleles) != 2 or alleles[0] == alleles[1] or "." in alleles:
                continue
            try:
                r, a = (int(x) for x in ad.split(",")[:2])
            except ValueError:
                continue
            ref_tot += r
            alt_tot += a
        total = ref_tot + alt_tot
        if total == 0:
            continue          # no het genotypes: nothing to test, not a failure
        frac = alt_tot / total
        if frac < AI_LOW or frac > AI_HIGH:
            bad.append(int(parts[0]))
    return np.array(bad, dtype=np.int64)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    chrom = sys.argv[1]

    # Use the RENAMED VCF (contig == chrom). intermediates/filtered/*.raw.vcf.gz still carries the
    # GenBank accession, and a region/target file keyed on `chrom` silently matches nothing
    # against it -- measured: bcftools -T reported 0 sites removed with no error.
    # UNPHASED source, renamed to chrN. Phasing strips AD/DP, so the filters cannot run on
    # the phased file; the pipeline order is filter -> phase, never phase -> filter.
    src = ROOT / "intermediates" / "filtered" / f"{chrom}.renamed.vcf.gz"
    if not src.exists():
        raw = ROOT / "intermediates" / "filtered" / f"{chrom}.raw.vcf.gz"
        rn = ROOT / "intermediates" / f"rename_{chrom}.txt"
        acc = run(f"bcftools view -H {raw} | head -1 | cut -f1").strip()
        rn.write_text(f"{acc}\t{chrom}\n")
        run(f"bcftools annotate --rename-chrs {rn} {raw} -Oz -o {src}")
        run(f"bcftools index -t {src}")
    out_vcf = ROOT / "intermediates" / "filtered" / f"{chrom}.sitefiltered.vcf.gz"
    out_txt = ROOT / "intermediates" / "filter_positions" / f"{chrom}.txt"
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    total = int(run(f"bcftools index -n {src}").strip())
    gatk = np.array(
        run(f"bcftools view -H {src} -i '{GATK_EXPR}' | cut -f2").split(),
        dtype=np.int64,
    )
    ai = allelic_imbalance_failures(src)
    med = median_summed_depth(src)
    depth, dlo, dhi = depth_failures(src, med)
    removed = np.union1d(np.union1d(gatk, ai), depth)

    print(f"{chrom}: {total:,} input sites")
    print(f"  GATK hard filters   : {gatk.size:,} ({100 * gatk.size / total:.2f}%)")
    print(f"  allelic imbalance   : {ai.size:,} ({100 * ai.size / total:.2f}%)")
    print(f"  depth outside [{dlo:.0f}, {dhi:.0f}] (median {med:.0f})"
          f": {depth.size:,} ({100 * depth.size / total:.2f}%)")
    print(f"  union removed       : {removed.size:,} ({100 * removed.size / total:.2f}%)")
    print(f"  sum of parts        : {gatk.size + ai.size + depth.size:,}"
          f"  -> overlap {gatk.size + ai.size + depth.size - removed.size:,}")

    # Sites in inaccessible sequence are excluded from filter.txt: the workflow strips them
    # from filtered_positions anyway, so listing them would be a no-op.
    mask = ROOT / "resources" / "panTro6" / "mask" / f"{chrom}.mask.bed"
    # Restricting filter.txt to ACCESSIBLE positions is an optimisation, not a correctness
    # requirement -- the workflow strips inaccessible positions out of filtered_positions
    # itself. The mask depends on genmap, which runs in parallel with this step, so when it
    # is not yet built record everything and let scripts/restrict_filter_txt.py trim later.
    if mask.exists():
        length = {n: int(l) for n, l in
                  (x.split() for x in open(ROOT / "resources" / "panTro6" / "chrom.sizes"))}[chrom]
        inacc = np.zeros(length, dtype=bool)
        for line in open(mask):
            _, a, b = line.split()
            inacc[int(a):int(b)] = True
        recorded = removed[~inacc[removed - 1]]
        print(f"  of those, accessible: {recorded.size:,}  -> written to filter.txt")
    else:
        recorded = removed
        print(f"  mask not built yet -> filter.txt holds all {recorded.size:,} removed "
              f"positions; run scripts/restrict_filter_txt.py once the mask exists")

    np.savetxt(out_txt, recorded, fmt="%d")

    # Excess heterozygosity WITHIN subspecies, for the accessibility mask (§4). A site where
    # an entire subspecies is >90% heterozygous is collapsed-paralogue evidence about the
    # REGION, so build_mask.py pads these +/-50 bp. Never pooled: the panel is structured and
    # pooling gives heterozygote deficit (Wahlund), not excess.
    import csv as _csv
    sub = {r["sample_id"]: r["subspecies"] for r in
           _csv.DictReader(open(ROOT / "resources/samples/subspecies.tsv"), delimiter="\t")}
    order = run(f"bcftools query -l {src}").split()
    grp = [sub.get(x, "UNK") for x in order]
    groups = sorted({g for g in grp if g not in ("UNK", "verus x troglodytes")})
    idx = {g: [i for i, x in enumerate(grp) if x == g] for g in groups}
    het_flag = []
    for line in run(f"bcftools query -f '%POS[\t%GT]\n' {src}").splitlines():
        parts = line.split("\t")
        gts = parts[1:]
        for g in groups:
            tot = het = 0
            for i in idx[g]:
                a = gts[i].replace("|", "/").split("/")
                if len(a) != 2 or "." in a:
                    continue
                tot += 1
                if a[0] != a[1]:
                    het += 1
            if tot >= 4 and het / tot > 0.9:
                het_flag.append(int(parts[0]))
                break
    np.save(ROOT / "intermediates" / f"{chrom}_het_flags.npy", np.array(het_flag, dtype=np.int64))
    print(f"  excess-het sites    : {len(het_flag):,}  -> intermediates/{chrom}_het_flags.npy")

    # Drop the failing sites from the VCF.
    # Derive the contig name from the file rather than assuming it equals `chrom`.
    contig = run(f"bcftools view -H {src} | head -1 | cut -f1").strip()
    if contig != chrom:
        raise ValueError(
            f"{src} uses contig {contig!r}, expected {chrom!r}; a target file keyed on "
            f"{chrom!r} would silently match nothing"
        )
    excl = ROOT / "intermediates" / f"{chrom}.exclude.txt"
    with open(excl, "w") as fh:
        for p in removed:
            fh.write(f"{contig}\t{p}\n")
    run(f"bcftools view -T ^{excl} {src} -Oz -o {out_vcf}")
    run(f"bcftools index -t {out_vcf}")
    kept = int(run(f"bcftools index -n {out_vcf}").strip())
    print(f"  kept in VCF         : {kept:,}")
    assert kept == total - removed.size, f"expected {total - removed.size}, got {kept}"
    excl.unlink()
    print(f"  wrote {out_vcf}")
    print(f"  wrote {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
