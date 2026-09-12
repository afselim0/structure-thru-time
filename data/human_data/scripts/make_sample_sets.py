"""
Define the 8 sample sets written to `resources/hgdp/sets/{set}.csv`.

Seven regional sets take the samples in that region, three of them capped per population to
hold down SINGER's cost, which grows steeply with sample count. The eighth, ALL, is a
geographically even subsample: the 4 highest-coverage samples from each of the 54 populations.

Per-population caps (REGION_CAPS below):

    MIDDLE_EAST           25 per population    161 -> 98
    CENTRAL_SOUTH_ASIA    15 per population    197 -> 130
    EAST_ASIA             10 per population    223 -> 158

AFRICA, EUROPE, OCEANIA and AMERICA are uncapped; their populations are small enough already.
A cap only ever removes samples, so a population already at or below its cap is untouched --
Uygur (10) under the CENTRAL_SOUTH_ASIA cap of 15, for instance, and all fifteen EAST_ASIA
populations that already sit at 10 or fewer.

Wherever a cap binds, the retained samples are the ones with the highest coverage. Ties in
coverage are broken by sample id, so the selection is deterministic and reproducible.

ALL is deliberately drawn from the FULL metadata rather than from the capped regional pools,
so capping a region does not change which individuals represent it in ALL.

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

# Maximum samples kept per population, by region. Regions absent here are uncapped.
# Applied as a cap, not a quota: populations already at or below the limit keep every sample.
REGION_CAPS = {
    "MIDDLE_EAST": 25,
    "CENTRAL_SOUTH_ASIA": 15,
    "EAST_ASIA": 10,
}


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
        in_region = [r for r in rows if r["region"] == region]
        assert in_region, f"No samples found for region {region}"
        cap = REGION_CAPS.get(region)

        by_population = defaultdict(list)
        for row in in_region:
            by_population[row["population"]].append(row)

        members = []
        capped_pops = []
        for population, pop_rows in sorted(by_population.items()):
            if cap is not None and len(pop_rows) > cap:
                # Highest coverage first; sample id breaks ties so this is deterministic.
                ranked = sorted(
                    pop_rows, key=lambda r: (-float(r["coverage"]), r["sample_id"])
                )
                capped_pops.append((population, len(pop_rows), cap))
                pop_rows = ranked[:cap]
            members.extend(r["sample_id"] for r in pop_rows)

        members = sorted(members)
        write_set(region, members)
        summary.append((region, len(members), len(by_population)))

        for population, before, after in capped_pops:
            print(f"    {region}/{population}: {before} -> {after}")

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
