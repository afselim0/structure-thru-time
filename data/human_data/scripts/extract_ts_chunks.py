"""
Split merged tree sequences into per-chunk `.trees` files for CoaSTT.

singer-snakemake emits one whole-chromosome tree sequence per MCMC replicate
(`<chrom>/trees/<chrom>.<rep>.tsz`). CoaSTT instead wants one tree sequence per chunk, in a
flat directory, treating each file as an independent contig -- `CountCoalEvents` globs
`*.trees` and uses the filename as the contig id (`src/CountCoalEvents.py:141-151`).

This takes the FINAL posterior draw (the highest replicate index, i.e. the most burnt-in
sample) and cuts it at the chunk boundaries SINGER actually ran on, writing:

Only ONE draw is written per chunk, deliberately. CoaSTT treats every `.trees` file in the
directory as an independent contig, so writing several posterior draws of the same region
would enter that region into the likelihood more than once.

The draw is also checked against the burn-in floor implied by `singer-mcmc-samples` and
`singer-mcmc-burnin`. singer-snakemake applies that burn-in only to its statistics and
diagnostic plots -- `merge_chunks` writes a tree sequence for every draw, including
pre-convergence ones -- so this script is the last point at which an unconverged draw can be
stopped before it reaches CoaSTT.

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


def final_replicate(trees_dir, chrom, min_replicate=0):
    """
    Path to the highest-numbered replicate, which is the last posterior draw.

    `min_replicate` is the burn-in floor: the index of the first draw the chain is considered
    converged at. It is not used to choose the draw -- the highest is always the most burnt-in
    -- but to REFUSE a directory whose best draw is still inside burn-in, which is how a run
    with too short a chain would otherwise reach CoaSTT unnoticed.
    """
    reps = []
    for name in os.listdir(trees_dir):
        if name.startswith(f"{chrom}.") and name.endswith(".tsz"):
            middle = name[len(chrom) + 1:-len(".tsz")]
            if middle.isdigit():
                reps.append(int(middle))
    if not reps:
        raise FileNotFoundError(f"No {chrom}.<rep>.tsz in {trees_dir}")
    best = max(reps)
    if best < min_replicate:
        raise SystemExit(
            f"{chrom}: highest replicate is {best}, inside the burn-in region "
            f"(< {min_replicate}). Every draw here is pre-convergence and must not be used. "
            f"Check singer-mcmc-samples / singer-mcmc-burnin in drivers/singer_config.yaml."
        )
    return os.path.join(trees_dir, f"{chrom}.{best}.tsz"), best


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
    parser.add_argument("--config", default=None,
                        help="drivers/singer_config.yaml; used to derive the burn-in floor. "
                             "Default: the copy next to this script's parent directory.")
    args = parser.parse_args()

    # Burn-in floor, read from the same config the run used so the two cannot drift apart.
    # singer-snakemake applies singer-mcmc-burnin only to statistics and plots; the tree
    # sequences are written for every draw, so nothing upstream stops a pre-convergence draw
    # from reaching CoaSTT. This is where that is caught.
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "drivers", "singer_config.yaml",
    )
    min_replicate = 0
    if os.path.exists(config_path):
        import yaml
        cfg = yaml.safe_load(open(config_path))
        n = int(cfg["singer-mcmc-samples"])
        min_replicate = int(float(cfg["singer-mcmc-burnin"]) * n)
        print(f"burn-in floor: replicate >= {min_replicate} of {n} "
              f"(singer-mcmc-burnin {cfg['singer-mcmc-burnin']})")
    else:
        print(f"WARNING: no config at {config_path}; burn-in floor not enforced")

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
        tsz_path, rep = final_replicate(
            os.path.join(base, "trees"), chrom, min_replicate=min_replicate
        )
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
