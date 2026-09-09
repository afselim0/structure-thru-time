"""
Figure data: whole-genome embeddings for the three badly-behaved admixture examples.

The figure's top row draws the three demographies straight from their demes files
(that is drawing, so it stays in the plotting script); the bottom row shows each
example's whole-genome embeddings, which are collected here.

The three examples are Procrustes-aligned to the recent_bottleneck run so the
panels are comparable rather than differing by the optimizer's arbitrary rotation
of the latent space.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (OUTPUTS_DIR, add_src_to_path, find_output_file, out_dir,
                     report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "badmix_WG"

ADMIX_DIR = os.path.join(OUTPUTS_DIR, "simulations", "admix")
EXAMPLES = ["three_way_admixture", "ghost_admixture", "recent_bottleneck"]
# the frame the other two are aligned into
REFERENCE = "recent_bottleneck"

D, N_MIN, N_MAX, P = 2, 100, 1000000, 1


def example_data_dir(example):
    return os.path.join(ADMIX_DIR, example, "sim", "data")


def main():
    destination = out_dir(FIGURE)

    raw, labels = {}, {}
    for example in EXAMPLES:
        data_dir = example_data_dir(example)
        raw[example] = pd.read_parquet(find_output_file(
            data_dir, "embeddings", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
        pop_labels = pd.read_csv(require(os.path.join(data_dir, "pop_labels.csv")))
        labels[example] = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]

    frames = []
    for example in EXAMPLES:
        df = (raw[example] if example == REFERENCE
              else procrustes_align_trajectories_df(raw[REFERENCE], raw[example]))
        frames.append(df.assign(example=example))
    embeddings = pd.concat(frames, ignore_index=True)

    pop_labels = pd.concat(
        [df.assign(example=example) for example, df in labels.items()],
        ignore_index=True)

    # One canonical population -> colour map shared by every example. Each run's own
    # pop_colors.csv assigns the palette in a different population order (A is purple
    # in three_way/ghost but green in recent_bottleneck), so a population would
    # otherwise change colour between panels.
    pop_colors = pd.DataFrame({
        "population": ["A", "B", "C", "D"],
        "color": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
    })

    params = pd.DataFrame([
        {"parameter": "min_time", "value": 5},
        {"parameter": "max_time", "value": 5000},
        {"parameter": "xy_lims", "value": 1e6},
        {"parameter": "reference_example", "value": REFERENCE},
    ])

    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
