"""
Figure data: the schematic describing the embeddings model.

Computes the true pair coalescence rates for the sawtooth demography,
embeds them through time, and derives the pairwise distances between the embedded
samples -- the three quantities the schematic's panels draw:
  A) the rate through time
  B) the embedding at two example timepoints
  C) the trajectories through time in 3D
  D) the distances through time, against 1/rate

The focal pair the figure highlights, and the colours the samples are drawn in,
are picked here too so the figure only lays out what it is given.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEMES_DIR, add_src_to_path, out_dir, report, require, write_table

add_src_to_path()
from compute_rates_true import compute_true_rates, expand_true_rates
from preprocess_demography import preprocess_demography
from LatentSpaceEmbeddings import LatentSpaceEmbeddings

FIGURE = "model_description"

DEME_FILE = os.path.join(DEMES_DIR, "sawtooth.yaml")
MAX_TIME = 40000
DELTA_T = 10
N_TIMEPOINTS = MAX_TIME // DELTA_T + 1
N_SAMPLES_PER_POP = 20
N_MIN = 1000
N_MAX = 1000000
# the optimizer starts from a random layout; this seed is what makes the figure
# reproducible
INIT_SEED = 0

# the two timepoints panel B slices at, and which pair the figure follows: the
# focal pair is taken by rank from the pairs sorted by their distance at t=0
T1, T2 = 300, 600
FOCAL_PAIR_ORDER = 5

FOCAL_COLOR = "#000000"
GENERIC_COLOR = "#BEBEBE"


def compute_distances(embeddings_df):
    """Euclidean distance between every pair of embedded samples, at each timepoint."""
    pairs = embeddings_df.merge(embeddings_df, on="time", suffixes=("_1", "_2"))
    pairs = pairs[pairs["individual_id_1"] < pairs["individual_id_2"]]
    pairs["distance"] = np.sqrt(
        (pairs["X1_1"] - pairs["X1_2"]) ** 2 + (pairs["X2_1"] - pairs["X2_2"]) ** 2)
    return (pairs.set_index(["individual_id_1", "individual_id_2", "time"])["distance"]
            .unstack("time"))


def main():
    destination = out_dir(FIGURE)

    require(DEME_FILE)
    pop_labels, _ = preprocess_demography(DEME_FILE, n_samples_per_pop=N_SAMPLES_PER_POP)
    true_rates = compute_true_rates(DEME_FILE, MAX_TIME, N_TIMEPOINTS)
    expanded_true_rates = expand_true_rates(true_rates, pop_labels, phased=False)

    LSE = LatentSpaceEmbeddings(expanded_true_rates, N_min=N_MIN, N_max=N_MAX)
    np.random.seed(INIT_SEED)
    X_init = np.random.rand(LSE.num_samples, LSE.n_components, LSE.num_timepoints)
    LSE.compute_embeddings_through_time(X_init)
    embeddings_df = LSE.get_embeddings_df(id_label="individual_id")

    distances_df = compute_distances(embeddings_df)
    times = distances_df.columns.values.astype(int)

    # the focal pair: the FOCAL_PAIR_ORDER-th widest-separated pair at the first
    # timepoint, so the figure follows a pair that starts clearly apart
    distance_order = distances_df.iloc[:, 0].sort_values().index[::-1]
    focal_i, focal_j = distance_order[FOCAL_PAIR_ORDER]

    # every sample is generic grey except the two the figure follows
    sample_colors = (pop_labels[["individual_id"]].drop_duplicates()
                     .reset_index(drop=True))
    sample_colors["role"] = "generic"
    sample_colors.loc[sample_colors["individual_id"].isin([focal_i, focal_j]), "role"] = "focal"
    sample_colors["color"] = np.where(sample_colors["role"] == "focal",
                                      FOCAL_COLOR, GENERIC_COLOR)

    # panel A draws one representative rate curve; every pair shares it in this
    # single-population demography, so the first row is what the figure plots
    rates_through_time = pd.DataFrame({
        "time": times,
        "rate": np.asarray(true_rates.values[0], dtype=float),
    })

    mean_distances = pd.DataFrame({
        "time": times,
        "mean_distance": distances_df.mean(axis=0).values,
        "focal_distance": distances_df.loc[(focal_i, focal_j)].values,
    })

    params = pd.DataFrame([
        {"parameter": "max_time", "value": MAX_TIME},
        {"parameter": "delta_t", "value": DELTA_T},
        {"parameter": "n_samples_per_pop", "value": N_SAMPLES_PER_POP},
        {"parameter": "N_min", "value": N_MIN},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "init_seed", "value": INIT_SEED},
        {"parameter": "t1", "value": T1},
        {"parameter": "t2", "value": T2},
        {"parameter": "focal_pair_order", "value": FOCAL_PAIR_ORDER},
        {"parameter": "focal_i", "value": focal_i},
        {"parameter": "focal_j", "value": focal_j},
        {"parameter": "xy_lims", "value": 140000},
    ])

    written = [
        write_table(embeddings_df, os.path.join(destination, "embeddings.parquet")),
        write_table(distances_df, os.path.join(destination, "distances.parquet")),
        write_table(rates_through_time, os.path.join(destination, "true_rates.parquet")),
        write_table(mean_distances, os.path.join(destination, "mean_distances.parquet")),
        write_table(sample_colors, os.path.join(destination, "sample_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
