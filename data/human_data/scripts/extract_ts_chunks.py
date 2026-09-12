"""
Split merged tree sequences into per-chunk `.trees` files for CoaSTT.

singer-snakemake emits one whole-chromosome tree sequence per MCMC replicate
(`<chrom>/trees/<chrom>.<rep>.tsz`). CoaSTT instead wants one tree sequence per chunk, in a
flat directory, treating each file as an independent contig -- `CountCoalEvents` globs
`*.trees` and uses the filename as the contig id (`src/CountCoalEvents.py:141-151`).

This takes the FINAL posterior draw (the highest replicate index, i.e. the most burnt-in
sample) and cuts it at the chunk boundaries SINGER actually ran on, writing:

    ts_chunks/<chrom>_<start_pos>_<end_pos>.trees        e.g. chr22_10519382_11519382.trees

The chromosome keeps its `chr` prefix deliberately: CoaSTT turns each filename into a contig
id, and an identifier starting with a digit is awkward in Python, pandas column access and
most downstream tooling.

Chunk boundaries come from `<chrom>.chunks.p`, a `RateMap` written by chunk_chromosomes.py:594
where `rate == 1.0` marks a chunk that was kept and `0.0` one that was skipped. Its positions
are one-based (`one_shift_ratemap` prepends 0 and adds 1 to every window boundary), so chunk i
spans [position[i], position[i+1]) and we convert back to the tree sequence's zero-based
coordinates when slicing.

Only kept chunks are written: skipped ones carry no SINGER output and would be empty.

`keep_intervals` is used rather than `trim`, so each chunk keeps the parent chromosome's
coordinate system -- positions in the output still refer to their true chromosomal location,
which is what makes the filename's start/end meaningful and lets results be traced back.
"""

import argparse
import os

import numpy as np
import pickle
import tskit
import tszip


def final_replicate(trees_dir, chrom):
    """Path to the highest-numbered replicate, which is the last posterior draw."""
    reps = []
    for name in os.listdir(trees_dir):
        if name.startswith(f"{chrom}.") and name.endswith(".tsz"):
            middle = name[len(chrom) + 1:-len(".tsz")]
            if middle.isdigit():
                reps.append(int(middle))
    if not reps:
        raise FileNotFoundError(f"No {chrom}.<rep>.tsz in {trees_dir}")
    return os.path.join(trees_dir, f"{chrom}.{max(reps)}.tsz"), max(reps)


def chunk_intervals(chunks_path):
    """
    Kept chunks as zero-based [start, end) pairs.

    chunks.p is one-based (see module docstring), so subtract the shift. The leading
    coordinate inserted by one_shift_ratemap is dropped along with it.
    """
    ratemap = pickle.load(open(chunks_path, "rb"))
    position = np.asarray(ratemap.position)
    rate = np.asarray(ratemap.rate)
    kept = []
    for i in range(rate.size):
        if rate[i] != 1.0:            # 0.0 = skipped chunk, nan = the inserted offset
            continue
        start = max(int(position[i]) - 1, 0)
        end = int(position[i + 1]) - 1
        if end > start:
            kept.append((start, end))
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--singer-output", required=True,
                        help="a set's singer-snakemake_outputs/ directory")
    parser.add_argument("--out-dir", required=True, help="ts_chunks/ to write into")
    parser.add_argument("--chromosomes", nargs="*", default=None,
                        help="default: every chromosome directory present")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    chroms = args.chromosomes
    if chroms is None:
        chroms = sorted(
            d for d in os.listdir(args.singer_output)
            if os.path.isdir(os.path.join(args.singer_output, d, "trees"))
        )
    if not chroms:
        raise SystemExit(f"No chromosome directories with trees/ under {args.singer_output}")

    total = 0
    for chrom in chroms:
        base = os.path.join(args.singer_output, chrom)
        tsz_path, rep = final_replicate(os.path.join(base, "trees"), chrom)
        intervals = chunk_intervals(os.path.join(base, f"{chrom}.chunks.p"))

        ts = tszip.decompress(tsz_path)

        print(f"{chrom}: replicate {rep}, {len(intervals)} kept chunks")
        for start, end in intervals:
            chunk = ts.keep_intervals([[start, end]], simplify=False)
            out = os.path.join(args.out_dir, f"{chrom}_{start}_{end}.trees")
            chunk.dump(out)
            total += 1

    print(f"\nWrote {total} chunk tree sequences to {args.out_dir}")


if __name__ == "__main__":
    main()
