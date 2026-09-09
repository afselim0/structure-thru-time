"""
Figure data: mean coalescence rates in the Wohns human dataset (application figure).

Aggregates the per-pair rate estimates into the three things the figure draws:
  A) the mean within-population rate for YRI, CEU and CHB
  B) the mean between-population rate for each pair of them
  C) every individual pair's rate, as the grey background

Times are converted from generations to years here, using the generation time the
figure labels its axis with.

Requires the empirical rate estimates. The Wohns run has not been produced on this
machine -- there are no application outputs under results/outputs/ and no app
config in data/configs/app_configs/ -- so this script stops with a clear message
until that run exists. RATES_FILE points at the location the figure used; adjust
it if the run lands elsewhere.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (METADATA_DIR, REPO_ROOT, add_src_to_path, out_dir, report,
                     require, write_table)

add_src_to_path()
from plot_rates import compute_mean_rates

FIGURE = "mean_coal_rates"

# where the figure read its rates from, relative to the repo root
RATES_FILE = os.path.join(REPO_ROOT, "results", "figures", "data",
                          "rates.minWS70.maxWS4000.V1.FS1.0.maxT20000.nT5000.parquet")
POP_LABELS_FILE = os.path.join(METADATA_DIR, "wohns_pop_labels.csv")

# generations to years
GENERATION_TIME = 28

WITHIN_POPS = ["YRI", "CEU", "CHB"]
BETWEEN_PAIRS = [("YRI", "CEU"), ("YRI", "CHB"), ("CEU", "CHB")]


def main():
    destination = out_dir(FIGURE)

    rates_df = pd.read_parquet(require(RATES_FILE, "Wohns empirical rate estimates"))
    pop_labels = pd.read_csv(require(POP_LABELS_FILE))

    # keep only the individuals the rate table actually covers
    active_pairs = rates_df.index.values
    active_samples = np.unique([s for pair in active_pairs for s in pair])
    pop_labels = pop_labels[pop_labels["individual_id"].isin(active_samples)]

    # compute_mean_rates keys on individual_id when per_haplotype is False, which is
    # what this unphased figure wants
    mean_rates, populations = compute_mean_rates(rates_df, pop_labels, per_haplotype=False)
    populations = list(populations)

    times_years = rates_df.columns.astype(float) * GENERATION_TIME

    within = pd.DataFrame({"time": times_years})
    for pop in WITHIN_POPS:
        idx = populations.index(pop)
        within[pop] = mean_rates[idx, idx, :]

    between = pd.DataFrame({"time": times_years})
    for pop1, pop2 in BETWEEN_PAIRS:
        i, j = populations.index(pop1), populations.index(pop2)
        between[f"{pop1}-{pop2}"] = mean_rates[i, j, :]

    # panel C draws every pair; the columns are relabelled in years so the figure
    # can plot the frame directly against the same axis as the panels beside it
    all_rates = rates_df.copy()
    all_rates.columns = times_years

    params = pd.DataFrame([
        {"parameter": "generation_time", "value": GENERATION_TIME},
        {"parameter": "n_pairs", "value": len(rates_df)},
    ])

    written = [
        write_table(within, os.path.join(destination, "within_population_rates.parquet")),
        write_table(between, os.path.join(destination, "between_population_rates.parquet")),
        write_table(all_rates, os.path.join(destination, "all_pair_rates.parquet")),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
