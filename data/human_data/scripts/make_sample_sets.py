"""
Define the 8 sample sets written to `resources/hgdp/sets/{set}.csv`.

Seven regional sets take every sample in that region. The eighth, ALL, is a geographically
even subsample: the 4 highest-coverage samples from each of the 54 populations, 216 samples.
Taking all 929 would make SINGER's cost (quadratic in sample count) prohibitive, and an
unweighted subsample would be dominated by the densely sampled regions -- EAST_ASIA alone is
223 samples against OCEANIA's 28.

Every population has at least 4 samples, so the per-population quota is always met exactly.
Ties in coverage are broken by sample id so the selection is reproducible.

Each CSV is a headerless single column of sample ids, which is the format `bcftools view -S`
expects.
"""

import argparse
import csv
import os
from collections import defaultdict

PER_POPULATION = 4
REGIONS = [
    "AFRICA", "EUROPE", "CENTRAL_SOUTH_ASIA", "EAST_ASIA",
    "OCEANIA", "AMERICA", "MIDDLE_EAST",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="resources/hgdp/metadata.tsv")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    with open(args.metadata) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    os.makedirs(args.outdir, exist_ok=True)

    def write_set(name, sample_ids):
        path = os.path.join(args.outdir, f"{name}.csv")
        with open(path, "w") as out:
            for sample_id in sample_ids:
                out.write(f"{sample_id}\n")
        return path

    summary = []

    for region in REGIONS:
        members = sorted(r["sample_id"] for r in rows if r["region"] == region)
        assert members, f"No samples found for region {region}"
        write_set(region, members)
        summary.append((region, len(members), len({
            r["population"] for r in rows if r["region"] == region
        })))

    # ALL: highest-coverage `PER_POPULATION` per population.
    by_population = defaultdict(list)
    for row in rows:
        by_population[row["population"]].append(row)

    chosen = []
    for population, members in sorted(by_population.items()):
        assert len(members) >= PER_POPULATION, (
            f"Population {population} has only {len(members)} samples, "
            f"fewer than the {PER_POPULATION} required"
        )
        ranked = sorted(
            members, key=lambda r: (-float(r["coverage"]), r["sample_id"])
        )
        chosen.extend(r["sample_id"] for r in ranked[:PER_POPULATION])

    chosen = sorted(chosen)
    write_set("ALL", chosen)
    summary.append(("ALL", len(chosen), len(by_population)))

    for name, n_samples, n_populations in summary:
        print(f"{name:<20} {n_samples:>4} samples  {n_populations:>3} populations")


if __name__ == "__main__":
    main()
