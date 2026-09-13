#!/usr/bin/env python3
"""
Trim filter.txt to accessible positions once the mask exists.

filter_sites.py runs in parallel with the mask build, so it may have written every removed
position. The workflow strips inaccessible positions from filtered_positions anyway, so this
is tidying rather than a correctness fix -- but it keeps the file honest about what it means.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def main():
    chrom = sys.argv[1]
    txt = ROOT / "intermediates" / "filter_positions" / f"{chrom}.txt"
    mask = ROOT / "resources" / "panTro6" / "mask" / f"{chrom}.mask.bed"
    length = {n: int(l) for n, l in
              (x.split() for x in open(ROOT / "resources/panTro6/chrom.sizes"))}[chrom]
    pos = np.loadtxt(txt, dtype=np.int64, ndmin=1)
    inacc = np.zeros(length, dtype=bool)
    for line in open(mask):
        _, a, b = line.split()
        inacc[int(a):int(b)] = True
    keep = pos[~inacc[pos - 1]]
    np.savetxt(txt, keep, fmt="%d")
    print(f"{chrom}: {pos.size:,} -> {keep.size:,} accessible ({pos.size - keep.size:,} masked)")

if __name__ == "__main__":
    sys.exit(main())
