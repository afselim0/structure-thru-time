"""
Split and COMPLEMENT the HGDP accessibility mask into per-chromosome inaccessible intervals.

This is the most dangerous step in the preparation, because getting it backwards fails
silently and yields plausible-looking numbers.

The published `hgdp_wgs.20190516.mask.bed` is a POSITIVE mask: its intervals are the regions
that are accessible (its own README says so explicitly, and the intervals sum to the
2,252,286,208 bp the README quotes). singer-snakemake wants the opposite. In
`chunk_chromosomes.py`:

    bedmask = np.loadtxt(mask_file, usecols=[1, 2])
    for (a, b) in bedmask: bitmask[a:b] = True   # True == INACCESSIBLE
    ...
    filter_accessible = ~bitmask[positions - 1]

So intervals listed in `chrN.mask.bed` are the ones DROPPED. We therefore write the
complement. Note also that `usecols=[1, 2]` ignores the chromosome column entirely -- a
genome-wide BED would be silently read as one chromosome's intervals -- so the file must be
split per chromosome.

The complement is taken against the chromosome lengths in `chrom.sizes`, which are derived
from the Ensembl ancestral FASTA, so mask, hapmap and FASTA all agree on one length.
"""

import argparse
import os
import subprocess
import sys
import tempfile

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

# The accessible fraction of an autosome, measured: 48%-82% across the 22, mean ~74%.
#
# The lower end is set by the acrocentric chromosomes, whose p-arms are satellite/rDNA repeat
# and entirely unmappable -- chr22's single largest inaccessible interval is 0-10,661,594, the
# whole p-arm, which is why it sits lowest at 48%. chr21 (5.1 Mb p-arm) and chr15 (19.8 Mb)
# show the same structure. That is real biology, not a polarity error.
#
# This band is a smoke test for INVERTED polarity, not a tight quality check. An inverted file
# would put most chromosomes near 1 - 0.74 = 26%, far below 0.40, while leaving chr22 near 52%
# -- so the check catches an inversion on the other 21 even though chr22 alone would be
# ambiguous. The README's "~73% of the primary assembly" uses a different denominator (it
# includes chrX/chrY and unplaced scaffolds); do not tighten this band to match it.
MIN_ACCESSIBLE_FRACTION = 0.40
MAX_ACCESSIBLE_FRACTION = 0.95


def read_chrom_sizes(path):
    sizes = {}
    with open(path) as handle:
        for line in handle:
            name, length = line.split()[:2]
            sizes[name] = int(length)
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", required=True, help="hgdp_wgs.20190516.mask.bed (positive)")
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    sizes = read_chrom_sizes(args.chrom_sizes)
    os.makedirs(args.outdir, exist_ok=True)

    # Split the genome-wide positive mask by chromosome in one pass.
    accessible_bp = {c: 0 for c in AUTOSOMES}
    handles = {}
    tmpdir = tempfile.mkdtemp(prefix="hgdp_mask_")
    try:
        for chrom in AUTOSOMES:
            handles[chrom] = open(os.path.join(tmpdir, f"{chrom}.bed"), "w")

        with open(args.mask) as handle:
            for line in handle:
                chrom, start, end = line.split()[:3]
                if chrom not in handles:
                    continue  # chrX, chrY: not analysed
                handles[chrom].write(f"{chrom}\t{start}\t{end}\n")
                accessible_bp[chrom] += int(end) - int(start)

        for handle in handles.values():
            handle.close()

        failures = []
        for chrom in AUTOSOMES:
            assert chrom in sizes, f"{chrom} absent from chrom.sizes"
            length = sizes[chrom]

            # bedtools complement needs a genome file restricted to this chromosome,
            # otherwise it emits whole-chromosome intervals for every other contig.
            genome_path = os.path.join(tmpdir, f"{chrom}.genome")
            with open(genome_path, "w") as genome:
                genome.write(f"{chrom}\t{length}\n")

            out_path = os.path.join(args.outdir, f"{chrom}.mask.bed")
            with open(out_path, "w") as out:
                result = subprocess.run(
                    ["bedtools", "complement",
                     "-i", os.path.join(tmpdir, f"{chrom}.bed"),
                     "-g", genome_path],
                    stdout=out, stderr=subprocess.PIPE, text=True,
                )
            if result.returncode != 0:
                raise RuntimeError(f"bedtools complement failed for {chrom}: {result.stderr}")

            inaccessible_bp = 0
            max_coord = 0
            with open(out_path) as check:
                for line in check:
                    _, start, end = line.split()[:3]
                    inaccessible_bp += int(end) - int(start)
                    max_coord = max(max_coord, int(end))

            assert max_coord <= length, (
                f"{chrom}: complemented mask reaches {max_coord}, past length {length}"
            )
            assert inaccessible_bp + accessible_bp[chrom] == length, (
                f"{chrom}: complement does not partition the chromosome "
                f"({inaccessible_bp} + {accessible_bp[chrom]} != {length})"
            )

            fraction = accessible_bp[chrom] / length
            status = "ok"
            if not MIN_ACCESSIBLE_FRACTION <= fraction <= MAX_ACCESSIBLE_FRACTION:
                status = "OUT OF RANGE"
                failures.append((chrom, fraction))
            print(
                f"{chrom}\taccessible={fraction:6.2%}\t"
                f"inaccessible={inaccessible_bp:>11,} bp\t{status}"
            )

        if failures:
            print(
                "\nERROR: accessible fraction outside "
                f"[{MIN_ACCESSIBLE_FRACTION:.0%}, {MAX_ACCESSIBLE_FRACTION:.0%}] for: "
                + ", ".join(f"{c} ({f:.2%})" for c, f in failures),
                file=sys.stderr,
            )
            print(
                "This usually means the mask polarity is inverted -- the written files must "
                "list INACCESSIBLE intervals.",
                file=sys.stderr,
            )
            sys.exit(1)

        total_accessible = sum(accessible_bp.values())
        total_length = sum(sizes[c] for c in AUTOSOMES)
        print(
            f"\nAutosomes: {total_accessible:,} / {total_length:,} accessible "
            f"({total_accessible / total_length:.2%})"
        )
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for name in os.listdir(tmpdir):
            os.unlink(os.path.join(tmpdir, name))
        os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
