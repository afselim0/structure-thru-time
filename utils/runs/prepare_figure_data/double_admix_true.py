"""
Figure data: the two-admixture example under true local ancestry.

The figure shows the whole-genome embeddings and how well their reconstructed
rates match the input ones, plus one embeddings panel per source ancestry (A-D)
computed from the true local ancestry.

The four ancestry-specific embeddings are Procrustes-aligned before export: A and
B to the A frame, C and D to the D frame, matching which admixed population each
pair of ancestries contributes to (see data/demes/two_admix.yaml). The latent
space is only identified up to rotation, reflection and scale, so this is what
makes the panels visually comparable.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (OUTPUTS_DIR, add_src_to_path, find_output_file, out_dir,
                     report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "double_admix_true"

SIM_DIR = os.path.join(OUTPUTS_DIR, "simulations", "admix", "two_admix", "sim")
DATA_DIR = os.path.join(SIM_DIR, "data")

D, N_MIN, N_MAX, P = 2, 100, 1000000, 1
ANCESTRIES = ["A", "B", "C", "D"]
# A and B admix into Admix_1 and C and D into Admix_2, so each pair is aligned
# within its own frame
ALIGNMENT_REFERENCE = {"A": "A", "B": "A", "C": "D", "D": "D"}


def ancestry_data_dir(ancestry, method="True"):
    return os.path.join(SIM_DIR, "ancestry_STT", method, ancestry, "data")


def main():
    destination = out_dir(FIGURE)

    # ---- whole-genome embeddings and rates ----
    embeddings_WG = pd.read_parquet(
        find_output_file(DATA_DIR, "embeddings", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
    input_rates_WG = pd.read_parquet(find_output_file(DATA_DIR, "rates"))
    reconstructed_rates_WG = pd.read_parquet(find_output_file(
        DATA_DIR, "reconstructed_rates", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))

    # ---- per-ancestry embeddings, aligned within each admixed population's frame ----
    raw = {}
    for ancestry in ANCESTRIES:
        raw[ancestry] = pd.read_parquet(find_output_file(
            ancestry_data_dir(ancestry), "embeddings",
            d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))

    frames = []
    for ancestry in ANCESTRIES:
        reference_key = ALIGNMENT_REFERENCE[ancestry]
        df = (raw[ancestry] if ancestry == reference_key
              else procrustes_align_trajectories_df(
                  raw[reference_key], raw[ancestry], allow_reflections=True))
        frames.append(df.assign(ancestry=ancestry))
    ancestry_embeddings = pd.concat(frames, ignore_index=True)

    pop_labels = pd.read_csv(require(os.path.join(DATA_DIR, "pop_labels.csv")))
    pop_colors = pd.read_csv(require(os.path.join(DATA_DIR, "pop_colors.csv")))
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
    pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]
    # the last population's colour is overridden so the two admixed populations stay
    # distinguishable against the source populations in the figure
    pop_colors.iloc[-1, pop_colors.columns.get_loc("color")] = "#505011"

    params = pd.DataFrame([
        {"parameter": "min_time", "value": 50},
        {"parameter": "max_time", "value": 5000},
        {"parameter": "xy_lims", "value": 1e6},
        {"parameter": "N_max", "value": N_MAX},
    ])

    written = [
        write_table(embeddings_WG, os.path.join(destination, "embeddings_WG.parquet")),
        write_table(input_rates_WG, os.path.join(destination, "input_rates_WG.parquet")),
        write_table(reconstructed_rates_WG, os.path.join(destination, "reconstructed_rates_WG.parquet")),
        write_table(ancestry_embeddings, os.path.join(destination, "ancestry_embeddings.parquet")),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
