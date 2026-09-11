"""
Write `chr{N}.meta.csv` for one sample set, ordered to match the VCF.

This is a silent-failure hazard worth being explicit about. singer-snakemake reads the
metadata like this (`chunk_chromosomes.py`):

    metadata.append({k: v for k, v in zip(metadata_names, row)})
    assert len(metadata) == samples.size, "Must have a metadata row for each sample"

It checks the row COUNT and nothing else, then zips rows to VCF samples positionally. There is
no matching on sample id, so a file in the wrong order passes every assertion and mislabels
every sample's population -- which would corrupt the cross-coalescence results that are the
entire point of the analysis, with no error anywhere.

So the order is taken from `bcftools query -l` on the actual VCF, and asserted against it.
The content is identical across a set's 22 chromosomes (bcftools preserves sample order), but
a file is written per chromosome because the workflow looks one up per VCF stem.
"""

import argparse
import csv
import os
import subprocess

COLUMNS = ["sample_id", "population", "region", "longitude", "latitude", "coverage"]
AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def vcf_samples(path):
    result = subprocess.run(
        ["bcftools", "query", "-l", path],
        check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="resources/hgdp/metadata.tsv")
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args()

    with open(args.metadata) as handle:
        by_id = {row["sample_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    reference_order = None
    for chrom in AUTOSOMES:
        vcf = os.path.join(args.input_dir, f"{chrom}.vcf.gz")
        assert os.path.exists(vcf), f"Missing {vcf}"

        order = vcf_samples(vcf)
        if reference_order is None:
            reference_order = order
        else:
            assert order == reference_order, (
                f"{chrom} sample order differs from chr1; metadata written per chromosome "
                f"would be inconsistent"
            )

        missing = [s for s in order if s not in by_id]
        assert not missing, f"{chrom}: samples absent from metadata: {missing[:5]}"

        out_path = os.path.join(args.input_dir, f"{chrom}.meta.csv")
        with open(out_path, "w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(COLUMNS)
            for sample_id in order:
                row = by_id[sample_id]
                writer.writerow([row[c] for c in COLUMNS])

        # Read back and confirm the order survived the round trip.
        with open(out_path) as check:
            written = [r["sample_id"] for r in csv.DictReader(check)]
        assert written == order, (
            f"{chrom}: metadata order does not match VCF order after writing"
        )

    print(
        f"  wrote meta.csv for {len(AUTOSOMES)} chromosomes, "
        f"{len(reference_order)} samples in VCF order"
    )


if __name__ == "__main__":
    main()
