#!/usr/bin/env python3
"""
Build the ancestral-state FASTA for one chromosome from the human outgroup.

Parses the panTro6->hg38 chain directly rather than shelling out to liftOver, for two
reasons. It nets in panTro6 coordinates -- the space we are filling -- so every panTro6
base is covered at most once by construction instead of by policy. And liftOver's BED
output silently drops the strand column unless the input carries six, which would leave
minus-strand bases uncomplemented with no error raised.

Chain format, and the two traps in it:

  chain score tName tSize tStrand tStart tEnd qName qSize qStrand qStart qEnd id

  * When qStrand is '-', the q coordinates are expressed in REVERSE-COMPLEMENT space and
    the cursor must be WALKED in that space, converting per block:
        forward_start = qSize - (q_revcomp_cursor + size)
    Converting once at the chain header instead (qSize - qEnd) is correct only for the
    first block and drifts thereafter. Getting the base complement right while leaving the
    coordinate in minus space yields a FASTA that passes a length check and is wrong
    everywhere -- measured at 29.4% identity against the panTro6 reference, versus 95.8%
    once the per-block conversion is used.
  * The base itself must also be complemented on those blocks.

Here t = panTro6 (what we fill) and q = hg38 (where the base comes from).

Guard: identity between the projected human base and the panTro6 reference, computed
SEPARATELY for plus- and minus-strand blocks. Human-chimp divergence is ~1.2%, so both
must land near 98-99%. A strand error collapses the affected class toward ~25% with
mismatches concentrated in complementary pairs, so a per-strand gate localises the bug to
one branch instead of averaging it away.

Run from data/chimp_data:
    python3 scripts/build_ancestral_fasta.py chr21
"""

import gzip
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pysam

ROOT = Path(__file__).resolve().parent.parent
CHAIN = Path("/project/jnovembre/afselim/chimp_args/data/reference/panTro6ToHg38.over.chain")
HG38 = Path("/project/jnovembre/afselim/chimp_args/data/reference/hg38.fa")
PANTRO6_ALL = ROOT / "resources" / "raw" / "panTro6.fa.gz"
LINE = 60

COMP = str.maketrans("ACGTN", "TGCAN")
# Human-chimp divergence is ~1.2%; anything below this on either strand class is a bug,
# not biology. A strand error lands near 0.25.
MIN_IDENTITY = 0.90


def chrom_length(chrom):
    for line in open(ROOT / "resources" / "panTro6" / "chrom.sizes"):
        name, length = line.split()
        if name == chrom:
            return int(length)
    raise KeyError(f"{chrom} not in chrom.sizes")


def iter_blocks(chain_path, target):
    """
    Yield (t_start, q_start_forward, size, q_name, q_strand) for aligned blocks on `target`.

    q_start_forward is always a forward-strand hg38 coordinate: on minus-strand chains the
    stored q coordinates are in reverse-complement space and are converted here, once, so
    no caller has to remember to do it.
    """
    keep = False
    with open(chain_path) as fh:
        for line in fh:
            if line.startswith("chain"):
                f = line.split()
                t_name, t_start = f[2], int(f[5])
                q_name, q_size, q_strand, q_start, q_end = f[7], int(f[8]), f[9], int(f[10]), int(f[11])
                keep = t_name == target
                if keep:
                    tpos = t_start
                    # Cursor stays in the chain's own q space; minus chains are converted
                    # per block below, because a single conversion at the header is right
                    # only for the first block.
                    qpos = q_start
                    cur = (q_name, q_strand, q_size)
                continue
            if not keep:
                continue
            parts = line.split()
            if not parts:
                keep = False
                continue
            size = int(parts[0])
            q_name_c, q_strand_c, q_size_c = cur
            q_fwd = qpos if q_strand_c == "+" else q_size_c - (qpos + size)
            yield tpos, q_fwd, size, q_name_c, q_strand_c
            if len(parts) >= 3:
                dt, dq = int(parts[1]), int(parts[2])
                tpos += size + dt
                qpos += size + dq
            else:
                keep = False


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    chrom = sys.argv[1]
    length = chrom_length(chrom)

    anc = np.full(length, ord("N"), dtype=np.uint8)
    covered = np.zeros(length, dtype=bool)
    conflict = np.zeros(length, dtype=bool)

    hg = pysam.FastaFile(str(HG38))
    human_contigs = set(hg.references)

    stats = {"+": Counter(), "-": Counter()}
    blocks = 0

    for t_start, q_start, size, q_name, q_strand in iter_blocks(CHAIN, chrom):
        if q_name not in human_contigs or t_start >= length:
            continue
        size = min(size, length - t_start)
        if size <= 0:
            continue
        blocks += 1

        seq = hg.fetch(q_name, q_start, q_start + size).upper()
        if len(seq) < size:
            size = len(seq)
            if size <= 0:
                continue
        if q_strand == "-":
            # Reverse-complement: the block was walked in forward coordinates, so the
            # bases come back in the opposite order from the panTro6 run they fill.
            seq = seq.translate(COMP)[::-1]

        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        sl = slice(t_start, t_start + size)
        # A panTro6 base claimed by two chains is paralogous; an outgroup call there is a
        # coin flip, so mark it and blank it rather than letting the last writer win.
        clash = covered[sl] & (anc[sl] != arr)
        conflict[sl] |= clash
        write = ~covered[sl]
        anc[sl] = np.where(write, arr, anc[sl])
        covered[sl] |= True
        stats[q_strand]["bases"] += int(size)
        stats[q_strand]["blocks"] += 1

    anc[conflict] = ord("N")

    # --- the gate: per-strand identity against the panTro6 reference ---
    # Per-chromosome extract if one exists, else pull the record from the whole-genome
    # FASTA. The identity gate needs the panTro6 reference for THIS chromosome.
    per_chrom = ROOT / "resources" / "raw" / f"panTro6_{chrom}.fa"
    if per_chrom.exists():
        ref_text = "".join(l.strip() for l in open(per_chrom) if not l.startswith(">"))
    else:
        chunks, keep = [], False
        with gzip.open(PANTRO6_ALL, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    if keep:
                        break
                    keep = line[1:].split()[0] == chrom
                elif keep:
                    chunks.append(line.strip())
        ref_text = "".join(chunks)
    ref = np.frombuffer(ref_text.upper().encode("ascii"), dtype=np.uint8)
    if ref.size != length:
        raise ValueError(f"panTro6 reference is {ref.size} bases, expected {length}")

    print(f"{chrom}: {length:,} bases, {blocks:,} aligned blocks")
    failures = []
    # Per-strand coverage, so the identity gate can be evaluated on each class separately.
    strand_mask = {"+": np.zeros(length, dtype=bool), "-": np.zeros(length, dtype=bool)}
    for t_start, q_start, size, q_name, q_strand in iter_blocks(CHAIN, chrom):
        if q_name not in human_contigs or t_start >= length:
            continue
        size = min(size, length - t_start)
        if size > 0:
            strand_mask[q_strand][t_start:t_start + size] = True

    for strand in ("+", "-"):
        sel = strand_mask[strand] & (anc != ord("N")) & (ref != ord("N"))
        n = int(sel.sum())
        if n == 0:
            print(f"  {strand} strand: no projected bases")
            continue
        ident = float((anc[sel] == ref[sel]).sum()) / n
        print(f"  {strand} strand: {n:,} bases, identity to panTro6 reference = {ident:.4f}")
        if ident < MIN_IDENTITY:
            mism = anc[sel] != ref[sel]
            pairs = Counter(zip(ref[sel][mism][:20000].tobytes().decode(),
                                anc[sel][mism][:20000].tobytes().decode()))
            comp = sum(v for (a, b), v in pairs.items() if b == a.translate(COMP))
            failures.append(
                f"{strand} strand identity {ident:.4f} < {MIN_IDENTITY} "
                f"({comp}/{sum(pairs.values())} sampled mismatches are complementary pairs "
                f"-- that signature means the strand handling is wrong)"
            )

    n_count = int((anc == ord("N")).sum())
    n_conf = int(conflict.sum())
    print(f"  projected          : {int(covered.sum()):,} ({100 * covered.sum() / length:.2f}%)")
    print(f"  multiply-covered->N: {n_conf:,} ({100 * n_conf / length:.4f}%)")
    print(f"  left as N          : {n_count:,} ({100 * n_count / length:.2f}%)")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1

    out = ROOT / "resources" / "panTro6" / "ancestral" / f"{chrom}.ancestral.fa.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = anc.tobytes().decode("ascii")
    with gzip.open(out, "wt") as fh:
        fh.write(f">{chrom}\n")
        for i in range(0, length, LINE):
            fh.write(text[i:i + LINE] + "\n")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
