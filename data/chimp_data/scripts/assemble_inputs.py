#!/usr/bin/env python3
"""
Assemble the final singer-snakemake input directory for one chromosome.

The phased VCF is copied straight through. There is no merge-back step: non-biallelic-SNP
records were removed in step 1 and stay removed, so the file that reaches SINGER is 100%
biallelic SNPs, every genotype phased, with nothing the workflow has to discard.

Their coordinates are not lost -- filter_sites.py wrote them into <chrom>.filter.txt, which
is what scales adj_mu. That matters more than it looks: the workflow adds multiallelics to
filtered_positions itself, but only ones it can SEE in the VCF, and it will see none. Listing
them in filter.txt is what keeps the mutation-rate correction honest.
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

    # Straight copy: the phased file already holds every record that should reach SINGER.
    run(f"bcftools view {phased} -Oz -o {out}")
    run(f"bcftools index -t {out}")

    n_ph = int(run(f"bcftools index -n {phased}").strip())
    n_fl = int(run(f"bcftools index -n {filtered}").strip())
    n_out = int(run(f"bcftools index -n {out}").strip())
    # Every site-filtered record is biallelic by construction, so phasing must preserve the
    # count exactly. A mismatch means SHAPEIT5 dropped something, which would be silent.
    assert n_out == n_ph == n_fl, (
        f"{chrom}: assembled {n_out}, phased {n_ph}, site-filtered {n_fl} -- all must match")

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

    print(f"{chrom}: {n_out:,} biallelic phased SNPs assembled")
    return 0

if __name__ == "__main__":
    sys.exit(main())
