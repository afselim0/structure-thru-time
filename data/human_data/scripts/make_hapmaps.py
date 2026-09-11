"""
Convert the deCODE 2019 sex-averaged genetic map to per-chromosome HapMap files.

Reads the Science supplement `aau1043_datas3.gz` (Halldorsson et al. 2019) and writes
`resources/grch38/hapmap/chr{1..22}.hapmap` in the 4-column HapMap layout that
`msprime.RateMap.read_hapmap` expects.

singer-snakemake calls `RateMap.read_hapmap(path)` bare, so the defaults apply: position
comes from column 2, cumulative cM from column 4, and *the rate column is ignored entirely*
(rates are derived by differencing the cumulative map). The rate column is still written
correctly -- see `shift` below -- so the file is not misleading to anything else that reads it.

The output `sequence_length` is the last Position(bp) in the file, and because an ancestral
FASTA is also supplied, `chunk_chromosomes.py` asserts
`hapmap.sequence_length == <fasta length>` *exactly*. That is why every file gets a terminal
row at the full chromosome length.
"""

import argparse
import gzip
import os

import numpy as np

# deCODE source columns, after the 7 comment lines and the header row.
SOURCE_COLUMNS = ("Chr", "Begin", "End", "cMperMb", "cM")
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def read_decode(path):
    """
    Parse the deCODE map into {chrom: (end_bp, cum_cM, rate_cM_per_Mb)}.

    The file carries 7 `#` comment lines *and* a real column-header row, so both have to be
    skipped. `cM` is the cumulative map position at `End` (not at `Begin`), which is what
    makes the alignment to Position(bp) exact with no shift.
    """
    by_chrom = {}
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        header = None
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if header is None:
                header = tuple(fields)
                assert header == SOURCE_COLUMNS, (
                    f"Unexpected deCODE columns {header}, expected {SOURCE_COLUMNS}. "
                    f"The map format may have changed."
                )
                continue
            chrom, _begin, end, cm_per_mb, cm = fields
            if chrom not in by_chrom:
                by_chrom[chrom] = ([], [], [])
            by_chrom[chrom][0].append(int(end))
            by_chrom[chrom][1].append(float(cm))
            by_chrom[chrom][2].append(float(cm_per_mb))
    return {
        chrom: (np.array(e, dtype=np.int64), np.array(c), np.array(r))
        for chrom, (e, c, r) in by_chrom.items()
    }


def write_hapmap(chrom, end_bp, cum_cm, rate, chrom_length, out_path):
    """
    Write one chromosome in HapMap format, extended to `chrom_length`.

    Three adjustments to the raw deCODE rows:

    1. A leading row at position 1 with 0 cM. The deCODE map starts well inside the
       chromosome (chr1 begins at 1,431,813), so without this the map would not cover the
       start. read_hapmap treats a leading row at nonzero position with 0 cM as missing data
       and gives that interval a NaN rate, which makes SINGER skip the first chunk -- that is
       expected and harmless, but the row must exist for the file to parse cleanly.

    2. The rate column shifted down one row. deCODE's cMperMb is the rate *ending* at End
       (backward-looking), while read_hapmap's rate semantics are forward-looking: the rate on
       a row applies to the interval that *starts* there. read_hapmap ignores this column, but
       shifting makes the file correct under either reading rather than silently off by one.

    3. A terminal row at exactly `chrom_length`, carrying the final cumulative cM forward.
       This is what makes sequence_length match the ancestral FASTA exactly. The implied rate
       over that trailing stretch is zero, which is the honest statement: deCODE has no data
       past its last marker.
    """
    assert np.all(np.diff(end_bp) > 0), f"{chrom}: positions not strictly increasing"
    assert np.all(np.diff(cum_cm) >= 0), f"{chrom}: cumulative cM decreases"
    assert end_bp[-1] <= chrom_length, (
        f"{chrom}: map extends to {end_bp[-1]} beyond chromosome length {chrom_length}"
    )

    # Forward-looking rate: row i carries the rate for [pos_i, pos_{i+1}).
    shifted_rate = np.append(rate[1:], 0.0)

    if end_bp[0] > 1:
        positions = np.concatenate([[1], end_bp])
        cm_values = np.concatenate([[0.0], cum_cm])
        rates = np.concatenate([[rate[0]], shifted_rate])
    else:
        # deCODE starts well inside every chromosome, so this branch is not taken in
        # practice. Guarding it anyway: prepending position 1 unconditionally would create a
        # duplicate first position, and read_hapmap requires positions to strictly increase.
        positions, cm_values, rates = end_bp.copy(), cum_cm.copy(), shifted_rate.copy()

    if positions[-1] < chrom_length:
        positions = np.append(positions, chrom_length)
        cm_values = np.append(cm_values, cm_values[-1])
        rates[-1] = 0.0
        rates = np.append(rates, 0.0)

    # What read_hapmap requires of the file we are about to write.
    assert np.all(np.diff(positions) > 0), f"{chrom}: output positions not strictly increasing"
    assert np.all(np.diff(cm_values) >= 0), f"{chrom}: output cM decreases"
    assert positions[-1] == chrom_length, (
        f"{chrom}: output ends at {positions[-1]}, not {chrom_length}; "
        f"sequence_length would not match the ancestral FASTA"
    )

    with open(out_path, "w") as out:
        out.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
        for pos, rat, cm in zip(positions, rates, cm_values):
            out.write(f"{chrom}\t{int(pos)}\t{rat:.10g}\t{cm:.10g}\n")

    return positions[-1], cm_values[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode", required=True, help="aau1043_datas3(.gz)")
    parser.add_argument("--chrom-sizes", required=True, help="two-column chrom.sizes")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    sizes = {}
    with open(args.chrom_sizes) as handle:
        for line in handle:
            name, length = line.split()[:2]
            sizes[name] = int(length)

    os.makedirs(args.outdir, exist_ok=True)
    decode = read_decode(args.decode)

    missing = [c for c in AUTOSOMES if c not in decode]
    assert not missing, f"deCODE map is missing {missing}"

    for chrom in AUTOSOMES:
        end_bp, cum_cm, rate = decode[chrom]
        assert chrom in sizes, f"{chrom} absent from chrom.sizes"
        out_path = os.path.join(args.outdir, f"{chrom}.hapmap")
        length, total_cm = write_hapmap(
            chrom, end_bp, cum_cm, rate, sizes[chrom], out_path
        )
        print(f"{chrom}\tlength={length}\ttotal={total_cm:.2f} cM\trows={end_bp.size + 2}")


if __name__ == "__main__":
    main()
