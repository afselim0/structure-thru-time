"""
Normalise the HGDP sample metadata to `resources/hgdp/metadata.tsv`.

The published file (`hgdp_wgs.20190516.metadata.txt`) has 16 columns; we keep the six that
singer-snakemake or downstream CoaSTT plotting actually use. Note the source column holding
the sample identifier is named `sample`, not `sample_id` -- it is renamed here so every
downstream consumer sees one spelling.
"""

import argparse
import csv

# source column -> output column
COLUMNS = {
    "sample": "sample_id",
    "population": "population",
    "region": "region",
    "longitude": "longitude",
    "latitude": "latitude",
    "coverage": "coverage",
}

EXPECTED_SAMPLES = 929
EXPECTED_POPULATIONS = 54
EXPECTED_REGIONS = {
    "AFRICA", "EUROPE", "CENTRAL_SOUTH_ASIA", "EAST_ASIA",
    "OCEANIA", "AMERICA", "MIDDLE_EAST",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="hgdp_wgs.20190516.metadata.txt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.metadata) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [c for c in COLUMNS if c not in reader.fieldnames]
        assert not missing, (
            f"Metadata is missing expected columns {missing}. "
            f"Found: {reader.fieldnames}"
        )
        rows = list(reader)

    assert len(rows) == EXPECTED_SAMPLES, (
        f"Expected {EXPECTED_SAMPLES} samples, found {len(rows)}"
    )

    populations = {r["population"] for r in rows}
    regions = {r["region"] for r in rows}
    assert len(populations) == EXPECTED_POPULATIONS, (
        f"Expected {EXPECTED_POPULATIONS} populations, found {len(populations)}"
    )
    assert regions == EXPECTED_REGIONS, (
        f"Region labels differ from expected: {regions ^ EXPECTED_REGIONS}"
    )

    ids = [r["sample"] for r in rows]
    assert len(set(ids)) == len(ids), "Duplicate sample identifiers in metadata"

    for row in rows:
        float(row["coverage"])  # fail loudly here rather than during set selection

    with open(args.out, "w", newline="") as out:
        writer = csv.writer(out, delimiter="\t")
        writer.writerow(COLUMNS.values())
        for row in rows:
            writer.writerow([row[c] for c in COLUMNS])

    print(
        f"Wrote {len(rows)} samples, {len(populations)} populations, "
        f"{len(regions)} regions to {args.out}"
    )


if __name__ == "__main__":
    main()
