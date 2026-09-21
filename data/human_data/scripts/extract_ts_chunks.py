"""
Split merged tree sequences into per-chunk `.trees` files for CoaSTT.

singer-snakemake emits one whole-chromosome tree sequence per MCMC replicate
(`<chrom>/trees/<chrom>.<rep>.tsz`). CoaSTT instead wants one tree sequence per chunk, in a
flat directory, treating each file as an independent contig -- `CountCoalEvents` globs
`*.trees` and uses the filename as the contig id (`src/CountCoalEvents.py:141-151`).

By default this takes the FINAL posterior draw (the highest replicate index, i.e. the most
burnt-in sample) and cuts it at the chunk boundaries SINGER actually ran on, writing:

    ts_chunks/<chrom>_<start_pos>_<end_pos>.trees        e.g. chr22_10519382_11519382.trees

`--n-draws N` instead extracts the last N draws, each into its OWN subdirectory:

    ts_chunks/draw_<rep>/<chrom>_<start_pos>_<end_pos>.trees

One directory per draw, never one directory holding several. CoaSTT treats every `.trees`
file in a directory as an independent contig, so mixing draws would enter the same region
into the likelihood more than once -- the draws are repeat measurements of one genome, not
extra genome. Each subdirectory is a complete, self-contained CoaSTT input; point a run at
one of them and compare runs across them to see posterior uncertainty.

Draws are checked against the burn-in floor implied by `singer-mcmc-samples`,
`singer-mcmc-burnin` and the ~4,000-iteration floor from SINGER's supplementary figure 10.
singer-snakemake applies its burn-in only to statistics and diagnostic plots -- `merge_chunks`
writes a tree sequence for every draw, including pre-convergence ones -- so this script is the
last point at which an unconverged draw can be stopped before it reaches CoaSTT.

The check REFUSES when this script picks the draw itself (the default single-draw path), and
only WARNS when a caller names how many draws to take: `--n-draws 8` means eight draws even if
the floor would allow fewer. The floor is a rule of thumb measured on SINGER's own, smaller
test cases, and spread across draws is the reason to extract several in the first place.

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

# Burn-in SINGER needs before a draw is usable, in MCMC iterations. From SINGER's supplementary
# figure 10. Measured on their sample sizes, so treat it as a floor rather than a guarantee:
# this set's ARGs are larger than most of their test cases.
MIN_BURNIN_ITERATIONS = 4000


def final_replicates(trees_dir, chrom, min_replicate=0, n_draws=1):
    """
    Paths to the `n_draws` highest-numbered replicates -- the last posterior draws.

    `min_replicate` is the burn-in floor: the index of the first draw the chain is considered
    converged at. It is not used to choose the draws -- the highest are always the most
    burnt-in -- but to police them.

    How hard it polices depends on who asked. Left to itself (`n_draws == 1`) this REFUSES a
    draw inside burn-in, which is how a run with too short a chain would otherwise reach
    CoaSTT unnoticed. An explicit `n_draws` is instead taken at its word and only WARNS: the
    caller asking for N draws is asking for N draws, and the floor is a rule of thumb read off
    SINGER's supplementary figure 10 rather than a property of this chain. Posterior spread is
    the point of extracting several draws, and the early ones still carry it.

    Returned oldest first, so the draw subdirectories are written in increasing order.
    """
    reps = []
    for name in os.listdir(trees_dir):
        if name.startswith(f"{chrom}.") and name.endswith(".tsz"):
            middle = name[len(chrom) + 1:-len(".tsz")]
            if middle.isdigit():
                reps.append(int(middle))
    if not reps:
        raise FileNotFoundError(f"No {chrom}.<rep>.tsz in {trees_dir}")
    if len(reps) < n_draws:
        raise SystemExit(
            f"{chrom}: asked for {n_draws} draws but only {len(reps)} replicates exist in "
            f"{trees_dir}. Re-run the merge step, or lower --n-draws."
        )
    chosen = sorted(reps)[-n_draws:]
    if chosen[0] < min_replicate:
        usable = sum(1 for r in reps if r >= min_replicate)
        below = [r for r in chosen if r < min_replicate]
        if n_draws == 1:
            raise SystemExit(
                f"{chrom}: highest replicate is {chosen[0]}, inside the burn-in region "
                f"(< {min_replicate}). Every draw here is pre-convergence and must not be "
                f"used. Check singer-mcmc-samples / singer-mcmc-burnin in "
                f"drivers/singer_config.yaml."
            )
        print(
            f"  WARNING: {chrom}: {len(below)} of the {n_draws} requested draws are inside "
            f"the burn-in region (< {min_replicate}): {', '.join(str(r) for r in below)}. "
            f"Only {usable} draw(s) clear it. Extracting all {n_draws} as asked -- treat the "
            f"flagged draws as less converged than the rest."
        )
    return [(os.path.join(trees_dir, f"{chrom}.{r}.tsz"), r) for r in chosen]


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
    parser.add_argument("--n-draws", type=int, default=1,
                        help="extract the last N posterior draws, each into its own "
                             "<out-dir>/draw_<rep>/ subdirectory. Default 1, which writes the "
                             "final draw straight into <out-dir> as it always has.")
    parser.add_argument("--draw", type=int, default=None,
                        help="extract this ONE replicate index into <out-dir>/draw_<rep>/. "
                             "Used to fan the draws of --n-draws out over parallel jobs; the "
                             "burn-in floor is enforced exactly as it is for --n-draws.")
    args = parser.parse_args()

    if args.draw is not None and args.n_draws != 1:
        raise SystemExit("--draw and --n-draws are alternatives; pass only one.")
    if args.n_draws < 1:
        raise SystemExit(f"--n-draws must be at least 1, got {args.n_draws}")

    # Burn-in floor, read from the same config the run used so the two cannot drift apart.
    # singer-snakemake applies singer-mcmc-burnin only to statistics and plots; the tree
    # sequences are written for every draw, so nothing upstream stops a pre-convergence draw
    # from reaching CoaSTT. This is where that is caught.
    #
    # The check is on CHAIN POSITION, not draw index. Draw k sits at MCMC iteration
    # (k + 1) * thin, so what matters is whether that iteration clears the burn-in the chain
    # needs -- roughly 4,000 per SINGER's supplementary figure 10. Checking the index alone
    # would be meaningless: at `singer-mcmc-burnin: 0.0` the index floor is 0 and every draw
    # passes, including draw 0 of a chain that never converged.
    config_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "drivers", "singer_config.yaml",
    )
    min_replicate = 0
    if os.path.exists(config_path):
        import yaml
        cfg = yaml.safe_load(open(config_path))
        n = int(cfg["singer-mcmc-samples"])
        thin = int(cfg["singer-mcmc-thin"])
        # index floor from the configured burn-in fraction
        min_replicate = int(float(cfg["singer-mcmc-burnin"]) * n)
        # ...and the index whose iteration first clears MIN_BURNIN_ITERATIONS
        iteration_floor = -(-MIN_BURNIN_ITERATIONS // thin) - 1   # ceil division, 0-based
        if iteration_floor > min_replicate:
            min_replicate = iteration_floor
        last_iter = n * thin
        print(f"chain: {n} draws x thin {thin} = {last_iter:,} iterations; "
              f"keeping draws >= {min_replicate} (iteration >= {(min_replicate + 1) * thin:,})")
        if last_iter < MIN_BURNIN_ITERATIONS:
            raise SystemExit(
                f"Chain is only {last_iter:,} iterations, shorter than the ~"
                f"{MIN_BURNIN_ITERATIONS:,} burn-in SINGER's supplementary figure 10 indicates. "
                f"No draw from this run is usable; raise singer-mcmc-samples or "
                f"singer-mcmc-thin in {config_path}."
            )
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

    # Resolve every chromosome's draws BEFORE writing anything, and require them to agree.
    #
    # A killed merge job leaves one chromosome short of its top draw. Since this takes the
    # highest replicates present per chromosome, that chromosome would silently contribute a
    # different, less burnt-in posterior sample than the rest -- and the resulting ts_chunks/
    # directory would look complete. Re-run the inference step to fill the gap; it redoes only
    # the missing merges, not the inference behind them.
    #
    # With --draw the replicate is named outright, so there is nothing to resolve and nothing
    # to disagree about: every chromosome must simply HAVE that draw. The floor is reported
    # but not enforced, exactly as under --n-draws -- a named draw is a caller who has already
    # decided.
    resolved = {}
    if args.draw is not None:
        if args.draw < min_replicate:
            # Named outright, so honoured outright -- this is one task of an explicit
            # multi-draw extraction, and --n-draws warns rather than refuses for the same
            # reason. Say so once, loudly, and carry on.
            print(
                f"  WARNING: --draw {args.draw} is inside the burn-in region "
                f"(< {min_replicate}); extracting it as asked, but treat it as less "
                f"converged than the later draws."
            )
        for chrom in chroms:
            path = os.path.join(args.singer_output, chrom, "trees", f"{chrom}.{args.draw}.tsz")
            if not os.path.exists(path):
                raise SystemExit(
                    f"{chrom}: no replicate {args.draw} at {path}. That usually means a merge "
                    f"job was killed; re-run the inference step to fill the gap."
                )
            resolved[chrom] = [(path, args.draw)]
    else:
        for chrom in chroms:
            trees_dir = os.path.join(args.singer_output, chrom, "trees")
            resolved[chrom] = final_replicates(
                trees_dir, chrom, min_replicate=min_replicate, n_draws=args.n_draws)

        reps_seen = {tuple(r for _, r in v) for v in resolved.values()}
        if len(reps_seen) > 1:
            detail = "\n".join(
                f"    {c}: replicate(s) {', '.join(str(r) for _, r in v)}"
                for c, v in sorted(resolved.items())
            )
            raise SystemExit(
                f"Chromosomes disagree on their latest replicates:\n{detail}\n"
                "That usually means a merge job was killed. Re-run the inference step to fill "
                "the gap -- it redoes only the missing merges, not the inference behind them."
            )

    # One draw goes straight into out-dir, preserving the layout every existing config points
    # at. Several go into draw_<rep>/ subdirectories -- one self-contained CoaSTT input each,
    # never mixed, since CountCoalEvents would read two draws of one region as two contigs.
    # --draw writes a subdirectory too: it is one task of a multi-draw extraction.
    nested = args.n_draws > 1 or args.draw is not None

    total = 0
    for chrom in chroms:
        base = os.path.join(args.singer_output, chrom)
        intervals = chunk_intervals(os.path.join(base, f"{chrom}.chunks.p"))

        for tsz_path, rep in resolved[chrom]:
            out_dir = os.path.join(args.out_dir, f"draw_{rep}") if nested else args.out_dir
            os.makedirs(out_dir, exist_ok=True)

            ts = tszip.decompress(tsz_path)

            print(f"{chrom}: replicate {rep}, {len(intervals)} kept chunks -> {out_dir}")
            for start, end in intervals:
                chunk = ts.keep_intervals([[start, end]], simplify=False)
                out = os.path.join(out_dir, f"{chrom}_{start}_{end}.trees")
                chunk.dump(out)
                total += 1

    print(f"\nWrote {total} chunk tree sequences to {args.out_dir}")


if __name__ == "__main__":
    main()
