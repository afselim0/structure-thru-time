"""
Figure data: per-ancestry embeddings for the three badly-behaved admixture examples.

Two figures share this data (badmix_GNN.pdf and badmix_RFmix.pdf): each is a 3x3
grid whose rows are the ancestries carried by the admixed individuals (B, C, D)
and whose columns are the three examples.

Every panel is Procrustes-aligned to one reference panel so the trajectories are
comparable across ancestries and examples rather than differing by the optimizer's
arbitrary rotation of the latent space.

Each example ran its embedding with its own window-size settings, so the runs are
resolved by their embedding parameters rather than by a shared filename.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (OUTPUTS_DIR, add_src_to_path, find_output_file, out_dir,
                     report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "badmix_ancestry"

ADMIX_DIR = os.path.join(OUTPUTS_DIR, "simulations", "admix")
EXAMPLES = ["three_way_admixture", "ghost_admixture", "recent_bottleneck"]
ANCESTRIES = ["B", "C", "D"]
METHODS = ["GNN", "RFmix"]

D, N_MIN, N_MAX, P = 2, 100, 1000000, 1

# the panel every other panel is aligned to (ghost_admixture / B ancestry)
REFERENCE = ("ghost_admixture", "B")


def ancestry_data_dir(example, method, ancestry):
    return os.path.join(ADMIX_DIR, example, "sim", "ancestry_STT", method, ancestry, "data")


def main():
    destination = out_dir(FIGURE)
    written = []

    for method in METHODS:
        raw = {}
        for example in EXAMPLES:
            for ancestry in ANCESTRIES:
                path = find_output_file(
                    ancestry_data_dir(example, method, ancestry), "embeddings",
                    d=D, Nmin=N_MIN, Nmax=N_MAX, p=P)
                raw[(example, ancestry)] = pd.read_parquet(path)

        reference = raw[REFERENCE]
        frames = []
        for (example, ancestry), df in raw.items():
            # the reference panel itself is left unrotated
            aligned = (df if (example, ancestry) == REFERENCE
                       else procrustes_align_trajectories_df(reference, df))
            frames.append(aligned.assign(example=example, ancestry=ancestry))
        written.append(write_table(
            pd.concat(frames, ignore_index=True),
            os.path.join(destination, f"{method}_embeddings.parquet")))

    # labels are per-example: they map each individual to its source population
    label_frames = []
    for example in EXAMPLES:
        path = require(os.path.join(ADMIX_DIR, example, "sim", "data", "pop_labels.csv"))
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        label_frames.append(df.assign(example=example))
    pop_labels = pd.concat(label_frames, ignore_index=True)

    # One canonical population -> colour map shared by every example, for the same
    # reason as in badmix_WG: each run's own pop_colors.csv orders the palette
    # differently, so a population would change colour between panels.
    pop_colors = pd.DataFrame({
        "population": ["A", "B", "C", "D"],
        "color": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
    })

    # recent_bottleneck's structure is recent enough to need a shorter time axis
    example_params = pd.DataFrame([
        {"example": "three_way_admixture", "min_time": 50},
        {"example": "ghost_admixture", "min_time": 50},
        {"example": "recent_bottleneck", "min_time": 10},
    ])

    params = pd.DataFrame([
        {"parameter": "max_time", "value": 5000},
        {"parameter": "xy_lims", "value": 1e6},
        {"parameter": "reference_example", "value": REFERENCE[0]},
        {"parameter": "reference_ancestry", "value": REFERENCE[1]},
    ])

    written += [
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(example_params, os.path.join(destination, "example_params.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
