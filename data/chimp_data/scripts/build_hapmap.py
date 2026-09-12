#!/usr/bin/env python3
"""
Build the 4-column hapmap for one chromosome from the Stevison ellioti map.

Route: the map's native hg18 coordinates -> hg38 -> panTro6. Passing through human is
native to the map's construction (it was built with reciprocal hg18 liftover), and there is
no direct chain from any earlier chimp assembly into panTro6 except panTro5.

Three traps in the source file, all of which chr21 happens to avoid -- which is exactly why
this script must be exercised on chr12 (100,476 minus1 rows, 311 na rows) and chr19 (404 na
rows) before a 23-chromosome run:

  * Coordinates are MEGABASES AS DECIMALS, not bp. Multiply by 1e6 first.
  * The minus token is spelled `minus1`, not `minus`. On those rows the SPECIES columns stay
    ascending but the HG18 columns descend, so the hg18 interval must be swapped before it
    is handed to liftOver.
  * `filter != 0` means rho/kb is the literal string "na" -- an interval where no rate was
    estimated, to be gap-filled at the genome average, never parsed as a number.

Normalisation uses ONE global scalar across all chromosomes so that small chromosomes keep
their genuinely higher cM/Mb; per-chromosome scaling would flatten that real signal.

Run from data/chimp_data:
    python3 scripts/build_hapmap.py chr12
"""

import gzip
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "resources" / "raw"
OUT = ROOT / "resources" / "panTro6" / "hapmap"
WORK = ROOT / "work"

# rho/kb -> cM/Mb, targeting a genome-wide mean of 1.1 cM/Mb. Measured across all 23
# autosomes: length-weighted mean rho/kb = 0.80044.
GLOBAL_SCALAR = 1.37424


def chrom_length(chrom):
    for line in open(ROOT / "resources" / "panTro6" / "chrom.sizes"):
        name, length = line.split()
        if name == chrom:
            return int(length)
    raise KeyError(f"{chrom} not in chrom.sizes")


def parse_source(chrom):
    """Read the map, returning (rows, n_minus1, n_na). Positions converted to bp."""
    src = RAW / f"{chrom}-map.hg18.txt.filtered.gz"
    rows = []
    n_minus1 = n_na = 0
    with gzip.open(src, "rt") as fh:
        fh.readline()
        for i, line in enumerate(fh):
            p = line.rstrip("\n").split("\t")
            if len(p) < 14:
                continue
            hg_s, hg_e = float(p[9]), float(p[10])
            if p[13] != "plus":
                n_minus1 += 1
                hg_s, hg_e = hg_e, hg_s      # hg18 columns descend on minus1 rows
            try:
                rho = float(p[5])
            except ValueError:
                rho = None                    # filter != 0: rho is the string "na"
                n_na += 1
            rows.append({
                "i": i,
                "hg_chrom": p[8],          # the map's own hg18 contig
                "hg_start": round(hg_s * 1e6),
                "hg_end": round(hg_e * 1e6),
                "rho": rho,
            })
    return rows, n_minus1, n_na


def liftover(chrom, rows):
    """hg18 -> hg38 -> panTro6, returning {row index: (start, end)} in panTro6 coordinates."""
    WORK.mkdir(exist_ok=True)
    bed = WORK / f"{chrom}_map_hg18.bed"
    with open(bed, "w") as fh:
        for r in rows:
            s, e = r["hg_start"], max(r["hg_end"], r["hg_start"] + 1)
            # Use the hg18 contig the map names. chr2A and chr2B both map to human chr2,
            # so writing the chimp name here matches nothing and lifts 0% of intervals.
            fh.write(f"{r['hg_chrom']}\t{s}\t{e}\t{r['i']}\n")

    step1 = WORK / f"{chrom}_map_hg38.bed"
    step2 = WORK / f"{chrom}_map_p6.bed"
    for src, chain, dst in [
        (bed, RAW / "hg18ToHg38.over.chain.gz", step1),
        (step1, RAW / "hg38ToPanTro6.over.chain.gz", step2),
    ]:
        subprocess.run(
            ["liftOver", str(src), str(chain), str(dst), str(dst) + ".unmap"],
            check=True, capture_output=True,
        )

    lifted = {}
    for line in open(step2):
        c, s, e, idx = line.rstrip("\n").split("\t")
        if c != chrom:
            continue          # a few intervals land on other chimp chromosomes; drop them
        lifted[int(idx)] = (int(s), int(e))
    return lifted


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    chrom = sys.argv[1]
    length = chrom_length(chrom)

    rows, n_minus1, n_na = parse_source(chrom)
    lifted = liftover(chrom, rows)
    print(f"{chrom}: {len(rows):,} source rows  minus1={n_minus1:,}  na={n_na:,}")
    print(f"  lifted onto {chrom}: {len(lifted):,} ({100 * len(lifted) / len(rows):.1f}%)")

    # Keep only intervals that lifted AND carry a rate; na rows become gaps.
    segs = sorted(
        (lifted[r["i"]][0], lifted[r["i"]][1], r["rho"])
        for r in rows if r["i"] in lifted and r["rho"] is not None
    )
    # liftOver jitter can produce overlaps; keep the first of any overlapping pair.
    clean = []
    for s, e, rho in segs:
        if e > s and (not clean or s >= clean[-1][1]):
            clean.append((s, e, rho))
    print(f"  usable non-overlapping intervals: {len(clean):,}")

    if not clean:
        raise ValueError(
            f"{chrom}: no map intervals survived liftover. Check that the hg18 contig in "
            f"the source file matches what the chain expects."
        )
    total_len = sum(e - s for s, e, _ in clean)
    avg_rho = sum(rho * (e - s) for s, e, rho in clean) / total_len
    print(f"  mean rho/kb {avg_rho:.4f} -> {avg_rho * GLOBAL_SCALAR:.4f} cM/Mb")

    # Cumulative map across the whole chromosome, gaps filled at the genome average so
    # filter_recombinant (rate > 0) does not drop those chunks entirely.
    pts = []
    cm = 0.0
    pos = 0
    for s, e, rho in clean:
        if s > pos:
            cm += avg_rho * GLOBAL_SCALAR * (s - pos) / 1e6
            pts.append((s, cm))
        cm += rho * GLOBAL_SCALAR * (e - s) / 1e6
        pts.append((e, cm))
        pos = e
    if pos < length:
        cm += avg_rho * GLOBAL_SCALAR * (length - pos) / 1e6
        pts.append((length, cm))

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{chrom}.hapmap"
    with open(dst, "w") as fh:
        fh.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
        fh.write(f"{chrom}\t1\t0.0\t0.0\n")
        for j, (p, c) in enumerate(pts):
            # Rate column shifted down one row: the rate FROM this position to the next.
            if j + 1 < len(pts):
                dp = pts[j + 1][0] - p
                rate = (pts[j + 1][1] - c) / (dp / 1e6) if dp > 0 else 0.0
            else:
                rate = 0.0
            fh.write(f"{chrom}\t{p}\t{rate:.6f}\t{c:.6f}\n")
    print(f"  total {cm:.2f} cM over {length / 1e6:.1f} Mb = {cm / (length / 1e6):.3f} cM/Mb")
    print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
