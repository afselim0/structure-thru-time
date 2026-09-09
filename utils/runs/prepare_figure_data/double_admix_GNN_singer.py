"""
Figure data: the two-admixture example under GNN local ancestry, on simulated
versus SINGER-inferred trees.

This is the SINGER counterpart of double_admix_GNN_RFmix.py. That figure held one
method per 3x4 grid; this one holds one tree source per grid, both under GNN local
ancestry: the columns are the four source ancestries and the rows are the
per-ancestry embeddings, the coalescence rates between that column's admixed
population and everyone else, and the input-vs-reconstructed rate histograms.

Alignment happens in two stages, so a SINGER panel is directly comparable with the
simulated-tree panel beside it:

  1. within the simulated-tree GNN run, each ancestry is put in its admixed
     population's frame exactly as the original prep does -- B onto A and C onto D,
     with A and D left as the optimizer produced them
  2. each SINGER ancestry is then aligned onto its own aligned simulated-tree
     counterpart

Rate limits are shared across both sources as well as all four ancestries, so every
rate and histogram panel of the figure sits on one scale.

Tables written to results/figure_data/double_admix_GNN_singer/:

  <source>_embeddings.parquet         long form with an `ancestry` column, aligned
  <source>_input_rates.parquet        wide rates, stacked with `ancestry` as the
  <source>_reconstructed_rates.parquet  outer index level
  rate_limits.csv                     source, rate_lo, rate_hi
  admix_populations.csv               ancestry, admix_pop
  pop_labels.csv, pop_colors.csv      labels and the shared palette
  params.csv                          parameter, value -- the plotting constants

`<source>` is the file-safe key "sim" or "singer"; sources.csv maps each to the
label the figure prints ("simulated trees" / "SINGER trees").
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (OUTPUTS_DIR, add_src_to_path, find_output_file, out_dir,
                     report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "double_admix_GNN_singer"

RUN_DIR = os.path.join(OUTPUTS_DIR, "simulations", "admix", "two_admix")
METHOD = "GNN"

# file-safe key -> the label the figure prints; the simulated-tree run is the frame
# everything is aligned into
SOURCES = {"sim": "simulated trees", "singer": "SINGER trees"}
REFERENCE_SOURCE = "sim"

D, N_MIN, N_MAX, P = 2, 100, 1000000, 1
ANCESTRIES = ["A", "B", "C", "D"]
# within a source, B is aligned onto A and C onto D: each pair of ancestries
# contributes to one admixed population, so they belong in one frame
ALIGNMENT_REFERENCE = {"A": "A", "B": "A", "C": "D", "D": "D"}
# ancestries A,B admix into Admix_1 and C,D into Admix_2 (data/demes/two_admix.yaml)
COL_ADMIX_POP = {"A": "Admix_1", "B": "Admix_1", "C": "Admix_2", "D": "Admix_2"}


def source_dir(source):
    return os.path.join(RUN_DIR, source)


def ancestry_data_dir(source, ancestry):
    return os.path.join(source_dir(source), "ancestry_STT", METHOD, ancestry, "data")


def read_source(source):
    """The embedding, input and reconstructed rates of every ancestry of one source."""
    embeddings, input_rates, reconstructed = {}, {}, {}
    for ancestry in ANCESTRIES:
        data_dir = ancestry_data_dir(source, ancestry)
        embeddings[ancestry] = pd.read_parquet(find_output_file(
            data_dir, "embeddings", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
        input_rates[ancestry] = pd.read_parquet(find_output_file(data_dir, "rates"))
        reconstructed[ancestry] = pd.read_parquet(find_output_file(
            data_dir, "reconstructed_rates", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
    return embeddings, input_rates, reconstructed


def align_within_source(embeddings):
    """Put B in A's frame and C in D's, as the simulated-tree figure does."""
    aligned = {}
    for ancestry in ANCESTRIES:
        reference_key = ALIGNMENT_REFERENCE[ancestry]
        aligned[ancestry] = (
            embeddings[ancestry] if ancestry == reference_key
            else procrustes_align_trajectories_df(
                embeddings[reference_key], embeddings[ancestry],
                allow_reflections=True))
    return aligned


def main():
    destination = out_dir(FIGURE)
    written = []

    labels_dir = os.path.join(source_dir(REFERENCE_SOURCE), "data")
    pop_labels = pd.read_csv(require(os.path.join(labels_dir, "pop_labels.csv")))
    pop_colors = pd.read_csv(require(os.path.join(labels_dir, "pop_colors.csv")))
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
    pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]
    # the run's last colour is too close to a neighbour's to tell the two apart in
    # the rate panels; overridden here as the original figure does
    pop_colors.iloc[-1, pop_colors.columns.get_loc("color")] = "#505011"

    raw, input_rates, reconstructed = {}, {}, {}
    for source in SOURCES:
        raw[source], input_rates[source], reconstructed[source] = read_source(source)

    # stage 1: the simulated-tree frames, aligned among themselves
    aligned = {REFERENCE_SOURCE: align_within_source(raw[REFERENCE_SOURCE])}
    # stage 2: each SINGER ancestry onto its own simulated-tree counterpart
    for source in SOURCES:
        if source == REFERENCE_SOURCE:
            continue
        aligned[source] = {
            ancestry: procrustes_align_trajectories_df(
                aligned[REFERENCE_SOURCE][ancestry], raw[source][ancestry],
                allow_reflections=True)
            for ancestry in ANCESTRIES}

    limit_rows = []
    for source in SOURCES:
        written.append(write_table(
            pd.concat([aligned[source][a].assign(ancestry=a) for a in ANCESTRIES],
                      ignore_index=True),
            os.path.join(destination, f"{source}_embeddings.parquet")))

        for name, tables in (("input_rates", input_rates[source]),
                             ("reconstructed_rates", reconstructed[source])):
            written.append(write_table(
                pd.concat(tables, names=["ancestry"]),
                os.path.join(destination, f"{source}_{name}.parquet")))

        # rate limits spanning every ancestry of this source, on the shifted input
        # rates the figure actually draws
        lo, hi = np.inf, -np.inf
        for ancestry in ANCESTRIES:
            x = (input_rates[source][ancestry] + 1 / N_MAX).values.ravel()
            y = reconstructed[source][ancestry].values.ravel()
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            lo = min(lo, x[valid].min(), y[valid].min())
            hi = max(hi, x[valid].max(), y[valid].max())
        limit_rows.append({"source": source, "rate_lo": lo, "rate_hi": hi})

    # one scale for the whole figure, so the two sources' panels are comparable
    shared = pd.DataFrame(limit_rows)
    shared["rate_lo"] = shared["rate_lo"].min()
    shared["rate_hi"] = shared["rate_hi"].max()

    sources = pd.DataFrame([{"source": key, "label": label}
                            for key, label in SOURCES.items()])
    admix_pops = pd.DataFrame(
        [{"ancestry": a, "admix_pop": COL_ADMIX_POP[a]} for a in ANCESTRIES])

    params = pd.DataFrame([
        {"parameter": "min_time", "value": 50},
        {"parameter": "max_time", "value": 5000},
        {"parameter": "xy_lims", "value": 1e6},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "time_scale", "value": "log"},
        {"parameter": "method", "value": METHOD},
    ])

    written += [
        write_table(shared, os.path.join(destination, "rate_limits.csv"), index=False),
        write_table(sources, os.path.join(destination, "sources.csv"), index=False),
        write_table(admix_pops, os.path.join(destination, "admix_populations.csv"), index=False),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
