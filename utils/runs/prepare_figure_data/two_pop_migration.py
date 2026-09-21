"""
Figure data: two populations splitting, with recent migration or a soft split.

For each of four scaled migration rates, builds two demographies -- one where the
populations exchange migrants only recently, one where they do so from the split
until part-way to the present ("soft split") -- computes their true pair
coalescence rates, and embeds them through time. The figure draws the resulting
trajectories as two rows of 3D panels.

The demes files these demographies are written to go into the output directory
alongside the tables, so the figure's inputs are all in one place.

Note on the time grid: compute_true_rates takes the time grid itself, built here
by rates_time_grid from the figure's delta_t. The plotting script this replaces
passed delta_t straight through as a point count, which yielded a single
timepoint and made LatentSpaceEmbeddings fail.
"""
import os
import sys

import numpy as np
import pandas as pd
import demes
import msprime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_src_to_path, out_dir, report, write_table

add_src_to_path()
from compute_rates_true import compute_true_rates, expand_true_rates, rates_time_grid
from preprocess_demography import preprocess_demography
from LatentSpaceEmbeddings import LatentSpaceEmbeddings

FIGURE = "two_pop_migration"

MAX_TIME = 3000
DELTA_T = 1
N_TIMEPOINTS = MAX_TIME // DELTA_T + 1
N_MAX = 1e6
N_MIN = 1
NE = 1e4
N_SAMPLES_PER_POP = 20
NM_LIST = np.array([1e-3, 1e-2, 1e-1, 1])
SPLIT_TIME = 2000
EARLY_MIG_END_TIME = 500
DEEP_MIG_END_TIME = 1500


def pop_split(N0, N1, N2, t_split, t_mig_start=0, t_mig_end=0, m=0):
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size=N0)
    demography.add_population(name="A", initial_size=N1, default_sampling_time=0)
    demography.add_population(name="B", initial_size=N2, default_sampling_time=0)
    demography.add_population_split(time=t_split, derived=["A", "B"], ancestral="Ancestral")
    if m != 0:
        demography.add_symmetric_migration_rate_change(
            time=t_mig_start, populations=['A', 'B'], rate=m)
        demography.add_symmetric_migration_rate_change(
            time=t_mig_end, populations=['A', 'B'], rate=0)
    demography.sort_events()
    return demography


def main():
    destination = out_dir(FIGURE)
    demes_dir = os.path.join(destination, "demes")
    os.makedirs(demes_dir, exist_ok=True)

    m_list = NM_LIST / NE
    frames = []
    labels_df = colors_df = None

    for Nm, m in zip(NM_LIST, tqdm(m_list)):
        early_file = os.path.join(demes_dir, f"early_migration_{m}.yaml")
        deep_file = os.path.join(demes_dir, f"deep_migration_{m}.yaml")
        demes.dump(pop_split(NE, NE, NE, SPLIT_TIME, t_mig_start=0,
                             t_mig_end=EARLY_MIG_END_TIME, m=m).to_demes(), early_file)
        demes.dump(pop_split(NE, NE, NE, SPLIT_TIME, t_mig_start=DEEP_MIG_END_TIME,
                             t_mig_end=SPLIT_TIME, m=m).to_demes(), deep_file)

        pop_labels, pop_colors = preprocess_demography(
            early_file, n_samples_per_pop=N_SAMPLES_PER_POP)
        # the two scenarios share one sampling scheme, so one copy of the labels
        # and colours describes every panel
        labels_df, colors_df = pop_labels, pop_colors

        for scenario, demes_file in (("early", early_file), ("deep", deep_file)):
            true_rates = compute_true_rates(demes_file, rates_time_grid(0, MAX_TIME, DELTA_T))
            true_rates = expand_true_rates(true_rates, pop_labels, phased=True)
            LSE = LatentSpaceEmbeddings(true_rates, N_min=N_MIN, N_max=N_MAX)
            LSE.compute_embeddings_through_time()
            frames.append(LSE.get_embeddings_df().assign(scenario=scenario, Nm=Nm, m=m))

    embeddings = pd.concat(frames, ignore_index=True)

    labels_df = labels_df.loc[:, ~labels_df.columns.str.startswith("Unnamed")]
    colors_df = colors_df.loc[:, ~colors_df.columns.str.startswith("Unnamed")]

    params = pd.DataFrame([
        {"parameter": "max_time", "value": MAX_TIME},
        {"parameter": "delta_t", "value": DELTA_T},
        {"parameter": "n_timepoints", "value": N_TIMEPOINTS},
        {"parameter": "Ne", "value": NE},
        {"parameter": "N_min", "value": N_MIN},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "n_samples_per_pop", "value": N_SAMPLES_PER_POP},
        {"parameter": "split_time", "value": SPLIT_TIME},
        {"parameter": "early_mig_end_time", "value": EARLY_MIG_END_TIME},
        {"parameter": "deep_mig_end_time", "value": DEEP_MIG_END_TIME},
    ])

    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(labels_df, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(colors_df, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
