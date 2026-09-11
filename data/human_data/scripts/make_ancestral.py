"""
Extract per-chromosome ancestral FASTAs and derive `chrom.sizes`.

Unpacks the Ensembl EPO 6-primate ancestral alleles tarball, takes
`homo_sapiens_ancestor_{1..22}.fa`, and rewrites each as a bgzipped single-record FASTA at
`resources/grch38/ancestral/chr{N}.ancestral.fa.gz`.

The lengths measured here become `resources/grch38/chrom.sizes`, the single source of truth
that the mask complement and the hapmap tail are both built against. singer-snakemake asserts
`hapmap.sequence_length == <this length>` exactly, so any disagreement stops the run.

Constraints imposed by `read_single_fasta` in singer-snakemake's utils.py:

  sequence = handle.read().strip().split("\n")
  headers = np.char.startswith(sequence, ">")
  assert len(headers) and headers[0] and np.all(~headers[1:])

- exactly one header line, and it must be first; no second record may follow
- the file is read whole and joined, so the sequence must start at position 1 and span the
  chromosome: indexing downstream is `ancfa[positions - 1]`

The EPO alphabet is ACGT (high confidence), acgt (low confidence), N (failure), - (indel) and
. (no coverage). `chunk_chromosomes.py` applies `np.char.upper`, so high- and low-confidence
calls are used identically; N/-/. fall through to "unpolarised" and are retained. We therefore
do no filtering on ancestral state here.
"""

import argparse
import gzip
import os
import subprocess
import sys
import tarfile

AUTOSOMES = list(range(1, 23))
VALID_BASES = set("ACGTNacgtn-.")


def extract_member(tar, member, dest):
    with tar.extractfile(member) as source, open(dest, "wb") as out:
        while True:
            block = source.read(1 << 22)
            if not block:
                break
            out.write(block)


def rewrite(fasta_path, chrom, out_path):
    """
    Rewrite one EPO FASTA as a bgzipped single-record file, returning its length.

    The header is replaced with the chromosome name. singer-snakemake ignores the header text
    (the bundled example just uses `>1`), but naming it after the chromosome keeps the file
    self-describing.
    """
    length = 0
    seen_header = False
    unexpected = set()

    bgzip = subprocess.Popen(
        ["bgzip", "-c"], stdin=subprocess.PIPE, stdout=open(out_path, "wb")
    )
    try:
        bgzip.stdin.write(f">{chrom}\n".encode())
        with open(fasta_path) as handle:
            for line in handle:
                if line.startswith(">"):
                    assert not seen_header, (
                        f"{fasta_path} holds more than one record; "
                        f"read_single_fasta requires exactly one"
                    )
                    seen_header = True
                    continue
                sequence = line.strip()
                if not sequence:
                    continue
                length += len(sequence)
                unexpected |= set(sequence) - VALID_BASES
                bgzip.stdin.write(f"{sequence}\n".encode())
    finally:
        bgzip.stdin.close()
        bgzip.wait()

    if bgzip.returncode != 0:
        raise RuntimeError(f"bgzip failed for {chrom}")
    assert seen_header, f"{fasta_path} has no FASTA header"
    if unexpected:
        print(
            f"  note: {chrom} carries unexpected characters {sorted(unexpected)}",
            file=sys.stderr,
        )
    return length


def verify(out_path, expected_length, chrom):
    """Re-read the written file the way singer-snakemake will, and confirm the length."""
    header_count = 0
    length = 0
    with gzip.open(out_path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                header_count += 1
            else:
                length += len(line.strip())
    assert header_count == 1, f"{chrom}: wrote {header_count} FASTA records, need exactly 1"
    assert length == expected_length, (
        f"{chrom}: re-read length {length} != {expected_length}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tarball", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--chrom-sizes", required=True)
    parser.add_argument("--workdir", required=True, help="scratch for unpacked FASTAs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    sizes = {}
    with tarfile.open(args.tarball, "r:gz") as tar:
        wanted = {
            f"homo_sapiens_ancestor_{n}.fa": n for n in AUTOSOMES
        }
        found = {}
        for member in tar:
            base = os.path.basename(member.name)
            if base in wanted and member.isfile():
                found[wanted[base]] = member

        missing = sorted(set(AUTOSOMES) - set(found))
        assert not missing, f"Tarball is missing ancestral FASTAs for chromosomes {missing}"

        for n in AUTOSOMES:
            chrom = f"chr{n}"
            out_path = os.path.join(args.outdir, f"{chrom}.ancestral.fa.gz")
            scratch = os.path.join(args.workdir, f"homo_sapiens_ancestor_{n}.fa")

            extract_member(tar, found[n], scratch)
            length = rewrite(scratch, chrom, out_path)
            verify(out_path, length, chrom)
            os.unlink(scratch)

            sizes[chrom] = length
            print(f"{chrom}\t{length:>12,} bp")

    with open(args.chrom_sizes, "w") as out:
        for n in AUTOSOMES:
            chrom = f"chr{n}"
            out.write(f"{chrom}\t{sizes[chrom]}\n")
    print(f"\nWrote {args.chrom_sizes}")


if __name__ == "__main__":
    main()
