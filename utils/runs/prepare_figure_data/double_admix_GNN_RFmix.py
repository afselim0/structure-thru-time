"""
Figure data: the two-admixture example under inferred local ancestry.

One figure per local-ancestry method (GNN and RFmix), each a 3x4 grid: the
per-ancestry embeddings, the coalescence rates between that column's admixed
population and everyone else, and the input-vs-reconstructed rate histograms.

Two pieces of data work happen here rather than in the figure:
  - the per-ancestry embeddings are Procrustes-aligned within each method, A and B
    to the A frame and C and D to the D frame, matching which admixed population
    each pair of ancestries contributes to
  - the shared log-log rate limits per method, spanning every ancestry's input
    (shifted by 1/N_max) and reconstructed rates, so the rate and histogram rows
    sit on exactly the same scale
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

FIGURE = "double_admix_GNN_RFmix"

SIM_DIR = os.path.join(OUTPUTS_DIR, "simulations", "admix", "two_admix", "sim")
DATA_DIR = os.path.join(SIM_DIR, "data")

D, N_MIN, N_MAX, P = 2, 100, 1000000, 1
ANCESTRIES = ["A", "B", "C", "D"]
METHODS = ["GNN", "RFmix"]
ALIGNMENT_REFERENCE = {"A": "A", "B": "A", "C": "D", "D": "D"}
# ancestries A,B admix into Admix_1 and C,D into Admix_2 (data/demes/two_admix.yaml)
COL_ADMIX_POP = {"A": "Admix_1", "B": "Admix_1", "C": "Admix_2", "D": "Admix_2"}


def ancestry_data_dir(method, ancestry):
    return os.path.join(SIM_DIR, "ancestry_STT", method, ancestry, "data")


def main():
    destination = out_dir(FIGURE)
    written = []

    pop_labels = pd.read_csv(require(os.path.join(DATA_DIR, "pop_labels.csv")))
    pop_colors = pd.read_csv(require(os.path.join(DATA_DIR, "pop_colors.csv")))
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
    pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]
    pop_colors.iloc[-1, pop_colors.columns.get_loc("color")] = "#505011"

    limit_rows = []
    for method in METHODS:
        raw, input_rates, reconstructed = {}, {}, {}
        for ancestry in ANCESTRIES:
            data_dir = ancestry_data_dir(method, ancestry)
            raw[ancestry] = pd.read_parquet(find_output_file(
                data_dir, "embeddings", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
            input_rates[ancestry] = pd.read_parquet(find_output_file(data_dir, "rates"))
            reconstructed[ancestry] = pd.read_parquet(find_output_file(
                data_dir, "reconstructed_rates", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))

        # ---- align each ancestry within its admixed population's frame ----
        frames = []
        for ancestry in ANCESTRIES:
            reference_key = ALIGNMENT_REFERENCE[ancestry]
            df = (raw[ancestry] if ancestry == reference_key
                  else procrustes_align_trajectories_df(
                      raw[reference_key], raw[ancestry], allow_reflections=True))
            frames.append(df.assign(ancestry=ancestry))
        written.append(write_table(
            pd.concat(frames, ignore_index=True),
            os.path.join(destination, f"{method}_embeddings.parquet")))

        # ---- rates, one table per method with the ancestry as a column level ----
        for name, tables in (("input_rates", input_rates),
                             ("reconstructed_rates", reconstructed)):
            combined = pd.concat(tables, names=["ancestry"])
            written.append(write_table(
                combined, os.path.join(destination, f"{method}_{name}.parquet")))

        # ---- shared rate limits across every ancestry of this method ----
        lo, hi = np.inf, -np.inf
        for ancestry in ANCESTRIES:
            x = (input_rates[ancestry] + 1 / N_MAX).values.ravel()
            y = reconstructed[ancestry].values.ravel()
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            lo = min(lo, x[valid].min(), y[valid].min())
            hi = max(hi, x[valid].max(), y[valid].max())
        limit_rows.append({"method": method, "rate_lo": lo, "rate_hi": hi})

    admix_pops = pd.DataFrame(
        [{"ancestry": a, "admix_pop": COL_ADMIX_POP[a]} for a in ANCESTRIES])

    params = pd.DataFrame([
        {"parameter": "min_time", "value": 50},
        {"parameter": "max_time", "value": 5000},
        {"parameter": "xy_lims", "value": 1e6},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "time_scale", "value": "log"},
    ])

    written += [
        write_table(pd.DataFrame(limit_rows), os.path.join(destination, "rate_limits.csv"), index=False),
        write_table(admix_pops, os.path.join(destination, "admix_populations.csv"), index=False),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
