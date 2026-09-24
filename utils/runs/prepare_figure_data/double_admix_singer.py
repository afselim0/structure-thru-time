"""
Figure data: the two-admixture example on SINGER-inferred trees, under GNN and
RFmix local ancestry.

Every SINGER embedding is put in the frame of the simulated-tree GNN run, so the
rows of this figure share one frame and match sim_figs/double_admix_GNN.pdf:

  1. within the simulated-tree GNN run, B is aligned onto A and C onto D
  2. each SINGER ancestry (both methods) is aligned onto its aligned
     simulated-tree counterpart

Tables written to results/figure_data/double_admix_singer/:

  embeddings.parquet        long form with `method` and `ancestry` columns, aligned
  input_rates.parquet       wide rates, stacked with (method, ancestry) outer levels
  rate_limits.csv           rate_lo, rate_hi shared by every rate panel
  admix_populations.csv     ancestry, admix_pop
  pop_labels.csv, pop_colors.csv
  params.csv                parameter, value
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

FIGURE = "double_admix_singer"

RUN_DIR = os.path.join(OUTPUTS_DIR, "simulations", "admix", "two_admix")
REFERENCE = ("sim", "GNN")
PANELS = [("singer", "GNN"), ("singer", "RFmix")]

D, N_MIN, N_MAX, P = 2, 100, 1000000, 1
ANCESTRIES = ["A", "B", "C", "D"]
ALIGNMENT_REFERENCE = {"A": "A", "B": "A", "C": "D", "D": "D"}
# ancestries A,B admix into Admix_1 and C,D into Admix_2 (data/demes/two_admix.yaml)
COL_ADMIX_POP = {"A": "Admix_1", "B": "Admix_1", "C": "Admix_2", "D": "Admix_2"}


def ancestry_data_dir(source, method, ancestry):
    return os.path.join(RUN_DIR, source, "ancestry_STT", method, ancestry, "data")


def read_run(source, method):
    embeddings, rates = {}, {}
    for ancestry in ANCESTRIES:
        data_dir = ancestry_data_dir(source, method, ancestry)
        embeddings[ancestry] = pd.read_parquet(find_output_file(
            data_dir, "embeddings", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
        rates[ancestry] = pd.read_parquet(find_output_file(data_dir, "rates"))
    return embeddings, rates


def main():
    labels_dir = os.path.join(RUN_DIR, REFERENCE[0], "data")
    pop_labels = pd.read_csv(require(os.path.join(labels_dir, "pop_labels.csv")))
    pop_colors = pd.read_csv(require(os.path.join(labels_dir, "pop_colors.csv")))
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
    pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]
    # as in double_admix_GNN_RFmix: the run's last colour is too close to a neighbour's
    pop_colors.iloc[-1, pop_colors.columns.get_loc("color")] = "#505011"

    runs = {panel: read_run(*panel) for panel in PANELS}
    destination = out_dir(FIGURE)

    ref_raw, _ = read_run(*REFERENCE)
    reference = {}
    for ancestry in ANCESTRIES:
        key = ALIGNMENT_REFERENCE[ancestry]
        reference[ancestry] = (ref_raw[ancestry] if ancestry == key
                               else procrustes_align_trajectories_df(
                                   ref_raw[key], ref_raw[ancestry], allow_reflections=True))

    frames, rate_tables = [], {}
    lo, hi = np.inf, -np.inf
    for (source, method), (embeddings, rates) in runs.items():
        for ancestry in ANCESTRIES:
            aligned = procrustes_align_trajectories_df(
                reference[ancestry], embeddings[ancestry], allow_reflections=True)
            frames.append(aligned.assign(method=method, ancestry=ancestry))
            rate_tables[(method, ancestry)] = rates[ancestry]
            x = (rates[ancestry] + 1 / N_MAX).values.ravel()
            x = x[np.isfinite(x) & (x > 0)]
            lo, hi = min(lo, x.min()), max(hi, x.max())

    admix_pops = pd.DataFrame(
        [{"ancestry": a, "admix_pop": COL_ADMIX_POP[a]} for a in ANCESTRIES])
    params = pd.DataFrame([
        {"parameter": "min_time", "value": 50},
        {"parameter": "max_time", "value": 5000},
        {"parameter": "xy_lims", "value": 1e6},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "time_scale", "value": "log"},
    ])

    written = [
        write_table(pd.concat(frames, ignore_index=True),
                    os.path.join(destination, "embeddings.parquet")),
        write_table(pd.concat(rate_tables, names=["method", "ancestry"]),
                    os.path.join(destination, "input_rates.parquet")),
        write_table(pd.DataFrame([{"rate_lo": lo, "rate_hi": hi}]),
                    os.path.join(destination, "rate_limits.csv"), index=False),
        write_table(admix_pops, os.path.join(destination, "admix_populations.csv"), index=False),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
