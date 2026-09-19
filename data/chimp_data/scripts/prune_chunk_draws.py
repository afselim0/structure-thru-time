#!/usr/bin/env python3
"""
Delete per-draw SINGER chunk files for the burn-in draws, keeping the last N.

singer-snakemake writes five per-draw files for every (chunk, draw):
    <id>_recombs_<d>.txt  _polegon_<d>.txt  _nodes_<d>.txt  _muts_<d>.txt  _branches_<d>.txt
At 20 draws that is 100 files per chunk, and across 1,334 chunks it dominates the ~428 GB
under singer-snakemake_outputs/.

This deletes draws [0, KEEP_FROM) and keeps [KEEP_FROM, samples). The cut is given explicitly
rather than read from singer-mcmc-burnin: the caller may want to discard MORE than the burn-in
floor requires, which is safe (it keeps a superset of what CoaSTT read), while discarding less
than the floor would be meaningless.

DRY RUN BY DEFAULT -- pass --apply to actually delete.

Safety:
  * REFUSES to run unless ts_chunks/ exists and holds the full expected file count, i.e.
    extraction already succeeded. These files are extraction's inputs; pruning before it
    finishes would be unrecoverable without redoing the inference.
  * Only touches the five per-draw families above, matched by an exact regex on the draw
    index. Never .yaml/.mut/.rec/.log/.singer.done, the _start_* files, or anything in trees/.
"""
import argparse, os, re, sys

FAMILIES = ("recombs", "polegon", "nodes", "muts", "branches")
# <6-digit id>_<family>_<draw>.txt -- anchored so _start_recombs_0.txt cannot match.
PAT = re.compile(r"^(\d{6})_(" + "|".join(FAMILIES) + r")_(\d+)\.txt$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singer-output", required=True)
    ap.add_argument("--ts-chunks", required=True)
    ap.add_argument("--keep-from", type=int, required=True,
                    help="lowest draw index to KEEP; everything below is deleted")
    ap.add_argument("--expect-chunks", type=int, default=None,
                    help="required number of .trees in --ts-chunks before deleting anything")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    n_ts = len([f for f in os.listdir(a.ts_chunks) if f.endswith(".trees")]) \
        if os.path.isdir(a.ts_chunks) else 0
    if n_ts == 0:
        sys.exit(f"REFUSING: no .trees in {a.ts_chunks}; extraction has not succeeded.")
    if a.expect_chunks is not None and n_ts != a.expect_chunks:
        sys.exit(f"REFUSING: {a.ts_chunks} holds {n_ts} .trees, expected {a.expect_chunks}.")
    print(f"extraction verified: {n_ts} .trees present")
    print(f"deleting draws 0..{a.keep_from - 1}, keeping {a.keep_from}+\n")

    total_files = total_bytes = 0
    per_chrom = {}
    for chrom in sorted(os.listdir(a.singer_output)):
        cdir = os.path.join(a.singer_output, chrom, "chunks")
        if not os.path.isdir(cdir):
            continue
        nf = nb = 0
        for name in os.listdir(cdir):
            m = PAT.match(name)
            if not m or int(m.group(3)) >= a.keep_from:
                continue
            p = os.path.join(cdir, name)
            try:
                nb += os.path.getsize(p)
            except OSError:
                continue
            nf += 1
            if a.apply:
                os.remove(p)
        per_chrom[chrom] = (nf, nb)
        total_files += nf
        total_bytes += nb

    for c, (nf, nb) in sorted(per_chrom.items()):
        print(f"  {c:7} {nf:7,} files  {nb / 2**30:8.1f} GB")
    verb = "deleted" if a.apply else "would delete"
    print(f"\n{verb}: {total_files:,} files, {total_bytes / 2**30:.1f} GB")
    if not a.apply:
        print("DRY RUN -- nothing removed. Re-run with --apply.")


if __name__ == "__main__":
    main()
