#!/usr/bin/env python3
"""
Assemble the final singer-snakemake input directory for one chromosome.

Merges the phased biallelic records back together with the multiallelic records that
SHAPEIT5 drops. Both are needed and for different reasons:

  * the phased biallelic records are what SINGER builds haplotypes from;
  * the multiallelic records must still be PRESENT so the workflow's own filter_biallelic
    sees them, drops them, and adds them to filtered_positions -- which is what scales
    adj_mu. Omitting them would inflate the inferred mutation rate by ~8% on chr21 and
    compress every date.

The multiallelic records keep their unphased genotypes; that is fine, because the workflow
discards them before any haplotype is built. They exist purely to be counted.
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "SingerSnakemake_Input"

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout

def main():
    chrom = sys.argv[1]
    IN.mkdir(exist_ok=True)
    filtered = ROOT / "intermediates" / "filtered" / f"{chrom}.sitefiltered.vcf.gz"
    phased   = ROOT / "intermediates" / "phased" / f"{chrom}.vcf.gz"
    out      = IN / f"{chrom}.vcf.gz"

    # Multiallelic records from the site-filtered (unphased) VCF.
    multi = ROOT / "intermediates" / f"{chrom}.multi.vcf.gz"
    run(f"bcftools view -m3 {filtered} -Oz -o {multi}")
    run(f"bcftools index -t {multi}")

    # Concatenate and re-sort: allow-overlaps merges the two coordinate streams.
    run(f"bcftools concat -a {phased} {multi} -Oz -o {out}")
    run(f"bcftools index -t {out}")
    multi.unlink(); Path(str(multi) + ".tbi").unlink()

    n_ph = int(run(f"bcftools index -n {phased}").strip())
    n_fl = int(run(f"bcftools index -n {filtered}").strip())
    n_out = int(run(f"bcftools index -n {out}").strip())
    assert n_out == n_fl, f"{chrom}: assembled {n_out} != site-filtered {n_fl}"

    # meta.csv in VCF order -- the workflow zips by index with no ID matching.
    order = run(f"bcftools query -l {out}").split()
    meta = {}
    import csv
    for r in csv.DictReader(open(ROOT / "resources/samples/subspecies.tsv"), delimiter="\t"):
        meta[r["sample_id"]] = r
    with open(IN / f"{chrom}.meta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "subspecies", "population", "mean_coverage"])
        for s in order:
            r = meta[s]
            w.writerow([s, r["subspecies"], r["population"], r["mean_coverage"]])

    # filter.txt: positions removed by the site filters, accessible ones only.
    src_txt = ROOT / "intermediates" / "filter_positions" / f"{chrom}.txt"
    (IN / f"{chrom}.filter.txt").write_text(src_txt.read_text())

    # Shared resources as symlinks.
    for suffix, sub in (("mask.bed", "mask"), ("hapmap", "hapmap"),
                        ("ancestral.fa.gz", "ancestral")):
        link = IN / f"{chrom}.{suffix}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / "resources" / "panTro6" / sub / f"{chrom}.{suffix}")

    print(f"{chrom}: phased {n_ph:,} + multiallelic {n_fl - n_ph:,} = {n_out:,} assembled")
    return 0

if __name__ == "__main__":
    sys.exit(main())
