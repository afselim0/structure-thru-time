"""
Define the 8 sample sets written to `resources/hgdp/sets/{set}.csv`.

Seven regional sets take the samples in that region, three of them capped per population to
hold down SINGER's cost, which grows steeply with sample count. The eighth, ALL, is a
geographically even subsample: the 4 highest-coverage samples from each of the 54 populations.

Per-population caps (REGION_CAPS below):

    MIDDLE_EAST           25 per population    161 -> 100
    CENTRAL_SOUTH_ASIA    15 per population    197 -> 130
    EAST_ASIA             10 per population    223 -> 168

Resulting set sizes: AFRICA 104, ALL 216, AMERICA 61, CENTRAL_SOUTH_ASIA 130, EAST_ASIA 168,
EUROPE 155, MIDDLE_EAST 100, OCEANIA 28.

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

Alongside the set definitions this also writes, per set, the two files CoaSTT needs to
label and colour a run, into `resources/population_labels_colors/`:

    <SET>_pop_labels.csv    one row per HAPLOTYPE, in tree-sequence sample order
    <SET>_pop_colors.csv    population -> colour, for the populations in that set

`sample_id` in pop_labels is the tree sequence's own sample (node) index and
`individual_id` its individual index, which is what CoaSTT indexes by; the HGDP_* columns
carry the original HGDP identifier so a row can be traced back to `metadata.tsv`.

The ORDER of pop_labels is the thing to be careful about. It is taken from
`SingerSnakemake_SampleSets/<SET>/chr1.meta.csv`, which write_set_metadata.py wrote in
`bcftools query -l` order, because that -- not the sorted set definition -- is the order
singer-snakemake fed to SINGER and therefore the order the samples appear in the tree
sequences. A tree sequence pairs individual i with nodes 2i and 2i+1, so haplotypes are
emitted two per individual in that same order. Getting this wrong would silently mislabel
every sample rather than raise, so the order is re-asserted against the tree sequences
themselves when `--trees-dir` is given.

Colours come from the two palettes written by HGDP_color_palette.py, which must be run
first. ALL uses `global_palette.csv`, whose colours encode which region a population is in;
the seven single-region sets use `regional_palette.csv`, whose colours are chosen to be
maximally distinct within a region. So a population has a different colour in ALL than in
its own region's set, on purpose -- the two plots ask different things of the colour.
"""

import argparse
import csv
import glob
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


LABEL_COLUMNS = [
    "sample_id", "individual_id", "HGDP_sample_id", "HGDP_individual_id",
    "population", "region",
]


def sample_order(set_name, sets_dir, fallback_ids):
    """HGDP sample ids for one set, in tree-sequence individual order.

    Prefers `<sets_dir>/<SET>/chr1.meta.csv` -- the VCF order SINGER actually saw. Falls
    back to the set definition's own order only when that file is absent, which is the
    case before the set's inputs have been built.
    """
    meta = os.path.join(sets_dir, set_name, "chr1.meta.csv") if sets_dir else None
    if meta and os.path.exists(meta):
        with open(meta) as handle:
            order = [row["sample_id"] for row in csv.DictReader(handle)]
        assert sorted(order) == sorted(fallback_ids), (
            f"{set_name}: chr1.meta.csv lists a different set of samples than "
            f"sets/{set_name}.csv"
        )
        return order, "chr1.meta.csv"
    return list(fallback_ids), "set definition (meta.csv absent)"


def verify_against_trees(set_name, order, trees_dir):
    """Assert the chosen order matches a real tree sequence. Returns True if checked."""
    pattern = os.path.join(trees_dir, set_name, "ts_chunks", "*.trees")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return False

    import tskit

    ts = tskit.load(matches[0])
    assert ts.num_individuals == len(order), (
        f"{set_name}: {matches[0]} has {ts.num_individuals} individuals, "
        f"expected {len(order)}"
    )
    in_trees = [ts.individual(i).metadata["sample_id"] for i in range(ts.num_individuals)]
    assert in_trees == order, (
        f"{set_name}: sample order does not match {os.path.basename(matches[0])}; "
        f"pop_labels would mislabel every sample"
    )
    for i in range(ts.num_individuals):
        assert list(ts.individual(i).nodes) == [2 * i, 2 * i + 1], (
            f"{set_name}: individual {i} does not hold nodes [{2 * i}, {2 * i + 1}]; "
            f"the haplotype numbering below assumes it does"
        )
    return True


def write_labels_and_colors(set_name, order, by_id, palette, outdir):
    """Write `<SET>_pop_labels.csv` (one row per haplotype) and `<SET>_pop_colors.csv`."""
    labels_path = os.path.join(outdir, f"{set_name}_pop_labels.csv")
    with open(labels_path, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for individual_id, hgdp_id in enumerate(order):
            row = by_id[hgdp_id]
            for sample_id in (2 * individual_id, 2 * individual_id + 1):
                writer.writerow({
                    "sample_id": sample_id,
                    "individual_id": individual_id,
                    "HGDP_sample_id": hgdp_id,
                    "HGDP_individual_id": hgdp_id,
                    "population": row["population"],
                    "region": row["region"],
                })

    # Populations in the order they first appear, so the colour file reads geographically
    # rather than alphabetically.
    populations = list(dict.fromkeys(by_id[i]["population"] for i in order))
    missing = [p for p in populations if p not in palette]
    assert not missing, (
        f"{set_name}: no colour in the palette for {missing}; "
        f"re-run HGDP_color_palette.py"
    )

    colors_path = os.path.join(outdir, f"{set_name}_pop_colors.csv")
    with open(colors_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["population", "color"])
        for population in populations:
            writer.writerow([population, palette[population]])

    return len(order), len(populations)


def read_palette(outdir, stem):
    path = os.path.join(outdir, f"{stem}.csv")
    assert os.path.exists(path), f"{path} not found; run HGDP_color_palette.py first"
    with open(path) as handle:
        return {r["population"]: r["color"] for r in csv.DictReader(handle)}


def write_all_labels_and_colors(set_members, rows, args):
    outdir = args.labels_outdir
    os.makedirs(outdir, exist_ok=True)

    # ALL plots every region together, so it takes the global palette, whose colours say
    # which region a population belongs to. The seven single-region sets plot one region
    # on its own, where that is wasted and telling populations apart is everything, so
    # they take the regional palette. A population therefore has a different colour in
    # ALL than in its own region's set -- deliberately; see HGDP_color_palette.py.
    palettes = {
        "global": read_palette(outdir, "global_palette"),
        "regional": read_palette(outdir, "regional_palette"),
    }

    by_id = {row["sample_id"]: row for row in rows}

    print()
    for set_name in sorted(set_members):
        order, source = sample_order(set_name, args.sets_dir, set_members[set_name])
        checked = (
            verify_against_trees(set_name, order, args.trees_dir)
            if args.trees_dir else False
        )
        which = "global" if set_name == "ALL" else "regional"
        n_individuals, n_populations = write_labels_and_colors(
            set_name, order, by_id, palettes[which], outdir
        )
        print(
            f"{set_name:<20} {2 * n_individuals:>4} haplotypes  "
            f"{n_individuals:>4} individuals  {n_populations:>3} populations  "
            f"{which:>8} palette  order from {source}"
            + ("  [verified against trees]" if checked else "")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="resources/hgdp/metadata.tsv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--labels-outdir",
                        help="resources/population_labels_colors; also holds palette.csv. "
                             "Omit to write only the set definitions.")
    parser.add_argument("--sets-dir",
                        help="SingerSnakemake_SampleSets/, for each set's chr1.meta.csv "
                             "(the authoritative sample order)")
    parser.add_argument("--trees-dir",
                        help="results/, to assert the order against real tree sequences")
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
    set_members = {}

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
        set_members[region] = members
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
    set_members["ALL"] = chosen
    summary.append(("ALL", len(chosen), len(by_population)))

    for name, n_samples, n_populations in summary:
        print(f"{name:<20} {n_samples:>4} samples  {n_populations:>3} populations")

    if args.labels_outdir:
        write_all_labels_and_colors(set_members, rows, args)


if __name__ == "__main__":
    main()
