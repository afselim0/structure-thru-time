#!/usr/bin/env python3
"""
Build the accessibility mask for one chromosome: INACCESSIBLE intervals.

Union of two signals:
  * genmap mappability < 1.0 at k=36, e=0 -- the operational criterion (can a read be placed
    uniquely?), not repeat annotation. genmap emits 1/(occurrence count), so `< 1.0` means
    "this 36-mer occurs more than once anywhere in the assembly": the strictest available
    criterion. k=36 e=0 was chosen by sweep -- k=100 e=2 left 97.85% accessible with a
    degenerate distribution no threshold could rescue, k=50 e=0 gave 97.29%, k=36 e=0 gives
    94.97%. Among tracks that say roughly the same thing, prefer the one that errs toward
    exclusion.
  * assembly gaps (UCSC gap track, keyed by RefSeq accession).

RepeatMasker is deliberately excluded: it covers 86.81% of chr21, and most Alu/MIR/L2
sequence is uniquely mappable and perfectly callable. Unique 36-mers are the operational
criterion. (The gap track adds 0 bp on top of mappability -- runs of N have no unique k-mers
-- which is the same point in miniature.)

Excess heterozygosity was REMOVED from this mask. It needed >= 4 called samples within a
subspecies to test, and the panel is too small per subspecies for that to be reliable
(ellioti n=5). A test that thin produces noise, not paralogue evidence.

Missingness is deliberately NOT included: the workflow strips inaccessible positions out of
filtered_positions, so masking a missing-genotype site would remove it from the mutation-rate
correction instead of adding information.
"""
import csv, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

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
    print(f"{chrom}: mappability {nm:,}  gaps {ng:,}  "
          f"-> inaccessible {tot:,} ({100*tot/L:.2f}%)  accessible {100*(L-tot)/L:.2f}%  "
          f"{len(starts):,} intervals")

if __name__ == "__main__":
    sys.exit(main())
