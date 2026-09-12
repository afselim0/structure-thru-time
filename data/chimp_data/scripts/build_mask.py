#!/usr/bin/env python3
"""
Build the accessibility mask for one chromosome: INACCESSIBLE intervals.

Union of three signals:
  * genmap mappability < 1.0 at k=36, e=0 -- the operational criterion (can a read be
    placed uniquely?), not repeat annotation. RepeatMasker is deliberately excluded: it
    covers ~87% of the assembly, and most Alu/MIR/L2 sequence is uniquely mappable.
  * assembly gaps (UCSC gap track, keyed by RefSeq accession)
  * excess heterozygosity WITHIN subspecies, padded +/-50 bp. Never pooled: the panel is
    structured, and pooling produces heterozygote deficit by the Wahlund effect.

Missingness is deliberately NOT included: the workflow strips inaccessible positions out of
filtered_positions, so masking a missing-genotype site would remove it from the mutation-rate
correction instead of adding information.
"""
import csv, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PAD = 50

def main():
    chrom = sys.argv[1]
    sizes = {n: int(l) for n, l in (x.split() for x in open(ROOT/"resources/panTro6/chrom.sizes"))}
    L = sizes[chrom]
    acc = {c: a for a, c in (x.split() for x in open(ROOT/"resources/panTro6/accession_to_chrom.txt"))}
    refseq = {}
    for r in csv.DictReader(open('/project/jnovembre/afselim/chimp_args/data/reference/panTro6_chrom_accessions.csv')):
        refseq[f"chr{r['chr_num']}"] = r['refseq_accession']

    inacc = np.zeros(L, dtype=bool)

    # 1. mappability
    bg = ROOT/"resources/raw/genmap_out/panTro6_autosomes.genmap.bedgraph"
    nm = 0
    for line in open(bg):
        c, s, e, v = line.split()
        if c != chrom:
            continue
        if float(v) < 1.0:
            a, b = int(s), min(int(e), L)
            inacc[a:b] = True
            nm += b - a
    # 2. assembly gaps
    key = refseq[chrom]
    ng = 0
    for line in open(ROOT/"resources/raw/panTro6.gap.bed"):
        p = line.split('\t', 4)
        if p[0] == key:
            a, b = int(p[1]), min(int(p[2]), L)
            inacc[a:b] = True
            ng += b - a
    # 3. excess heterozygosity within subspecies
    flags = ROOT/"work"/f"{chrom}_het_flags.npy"
    nh = 0
    if flags.exists():
        for p in np.load(flags):
            inacc[max(0, p-1-PAD):min(L, p+PAD)] = True
        nh = len(np.load(flags))

    out = ROOT/"resources/panTro6/mask"/f"{chrom}.mask.bed"
    out.parent.mkdir(parents=True, exist_ok=True)
    d = np.diff(inacc.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends = np.flatnonzero(d == -1) + 1
    if inacc[0]: starts = np.r_[0, starts]
    if inacc[-1]: ends = np.r_[ends, L]
    with open(out, 'w') as fh:
        for a, b in zip(starts, ends):
            fh.write(f"{chrom}\t{a}\t{b}\n")
    tot = int(inacc.sum())
    print(f"{chrom}: mappability {nm:,}  gaps {ng:,}  het-sites {nh:,}  "
          f"-> inaccessible {tot:,} ({100*tot/L:.2f}%)  accessible {100*(L-tot)/L:.2f}%  "
          f"{len(starts):,} intervals")

if __name__ == "__main__":
    sys.exit(main())
