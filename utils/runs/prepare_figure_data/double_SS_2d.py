"""
Figure data: the two-group 2D stepping-stone example.

The figure shows the smoothed embeddings through time, and at four timepoints how
the reconstructed rates compare with the estimated ones. The rate tables are
exported with the 1/N_max shift the figure draws them on already applied, so the
plotting script only has to scatter them.

Requires the *smoothed* embedding outputs (temporal smoothing penalty
l1 = [1e-2, 1e-3]). The double_2d_SS run currently on disk was produced with the
l1 line commented out in data/configs/sim_configs/double_2d_SS.yaml, so it holds
only the unsmoothed L1[None] outputs and this script will stop with a clear
message. Re-run the pipeline for that config with l1 enabled to produce them.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (OUTPUTS_DIR, find_output_file, out_dir, report, require,
                     write_table)

FIGURE = "double_SS_2d"

RUN_DIR = os.path.join(OUTPUTS_DIR, "simulations", "double_2d_SS", "sim", "data")

D, N_MIN, N_MAX_EMB, P = 2, 1, 10000000, 2
# the rate panels are drawn shifted by 1/N_max, with this N_max rather than the
# embedding's, exactly as the figure did
N_MAX_RATES = 1e6
# the four timepoints the bottom panels slice at
PANEL_TIMES = [150, 1500, 4500, 5400]


def main():
    destination = out_dir(FIGURE)

    embeddings = pd.read_parquet(find_output_file(
        RUN_DIR, "smooth_embeddings", d=D, Nmin=N_MIN, Nmax=N_MAX_EMB, p=P))
    estimated_rates = pd.read_parquet(find_output_file(RUN_DIR, "rates")) + 1 / N_MAX_RATES
    reconstructed_rates = pd.read_parquet(find_output_file(
        RUN_DIR, "smooth_reconstructed_rates",
        d=D, Nmin=N_MIN, Nmax=N_MAX_EMB, p=P)) + 1 / N_MAX_RATES

    if not np.allclose(estimated_rates.columns.astype(float),
                       reconstructed_rates.columns.astype(float)):
        raise ValueError("Estimated and reconstructed rates are on different time grids")
    if not estimated_rates.index.equals(reconstructed_rates.index):
        raise ValueError("Estimated and reconstructed rates cover different sample pairs")

    pop_labels = pd.read_csv(require(os.path.join(RUN_DIR, "pop_labels.csv")))
    pop_colors = pd.read_csv(require(os.path.join(RUN_DIR, "pop_colors.csv")))
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
    pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]

    # each rate pair is within one group or between the two; the figure colours the
    # scatter by this, so it is derived here rather than re-derived while plotting
    individual_groups = (
        pop_labels.drop_duplicates("individual_id")
        .set_index("individual_id")["population"].str.split("_").str[0])
    group_1 = individual_groups.reindex(estimated_rates.index.get_level_values(0)).to_numpy()
    group_2 = individual_groups.reindex(estimated_rates.index.get_level_values(1)).to_numpy()
    if pd.isna(group_1).any() or pd.isna(group_2).any():
        raise ValueError("Some rate pairs reference individuals missing from pop_labels")
    pair_types = pd.DataFrame(
        {"pair_type": np.where(group_1 == group_2, group_1, "between")},
        index=estimated_rates.index)

    params = pd.DataFrame([
        {"parameter": "N_max_rates", "value": N_MAX_RATES},
        {"parameter": "panel_times", "value": ",".join(str(t) for t in PANEL_TIMES)},
    ])

    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(estimated_rates, os.path.join(destination, "estimated_rates.parquet")),
        write_table(reconstructed_rates, os.path.join(destination, "reconstructed_rates.parquet")),
        write_table(pair_types, os.path.join(destination, "pair_types.parquet")),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
