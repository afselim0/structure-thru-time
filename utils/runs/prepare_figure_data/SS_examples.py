"""
Figure data: the 1D and 2D stepping-stone examples.

Two figures (SS_1d and SS_2d) share this script. Each shows six embeddings -- two
values of the link exponent p by three migration rates -- so twelve runs are read
in total.

Within each figure and each p, every migration rate's trajectories are
Procrustes-aligned to one reference rate. The latent space is only identified up to
rotation, reflection and scale, so without this the rows would differ by an
arbitrary transform rather than by the demography. The alignment is data work, so
it happens here and the figure reads the aligned coordinates.

Each figure's six embeddings go into one parquet keyed by (p, m); the per-figure
population labels and colours go alongside as csv.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (OUTPUTS_DIR, add_src_to_path, find_output_file, out_dir,
                     report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "SS_examples"

SS_DIMENSIONS = [1, 2]
MIGRATION_RATES = [0.05, 0.01, 0.001]
P_VALUES = [1, 2]

# each figure's rows are aligned to the migration rate that shows it most clearly
ALIGN_TO = {1: 0.01, 2: 0.001}
# the run each figure takes its labels and colours from
LABEL_RATE = 0.05


def run_data_dir(ss, m):
    """Data directory of one stepping-stone run, e.g. SS_1d_N100_m05."""
    m_mantissa = str(m).split(".")[1]
    return os.path.join(OUTPUTS_DIR, "simulations", "SS",
                        f"SS_{ss}d_N100_m{m_mantissa}", "sim", "data")


def main():
    destination = out_dir(FIGURE)
    written = []

    for ss in SS_DIMENSIONS:
        name = f"SS_{ss}d"

        # ---- load every p/m combination for this figure ----
        embeddings = {}
        for p in P_VALUES:
            for m in MIGRATION_RATES:
                path = find_output_file(run_data_dir(ss, m), "embeddings", p=p)
                embeddings[(p, m)] = pd.read_parquet(path)

        # ---- align each rate to the reference rate, within each p ----
        align_m = ALIGN_TO[ss]
        frames = []
        for p in P_VALUES:
            reference = embeddings[(p, align_m)]
            for m in MIGRATION_RATES:
                df = (reference if m == align_m
                      else procrustes_align_trajectories_df(reference, embeddings[(p, m)]))
                frames.append(df.assign(p=p, m=m))
        aligned = pd.concat(frames, ignore_index=True)

        label_dir = run_data_dir(ss, LABEL_RATE)
        pop_labels = pd.read_csv(require(os.path.join(label_dir, "pop_labels.csv")))
        pop_colors = pd.read_csv(require(os.path.join(label_dir, "pop_colors.csv")))
        pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
        pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]

        written += [
            write_table(aligned, os.path.join(destination, f"{name}_embeddings.parquet")),
            write_table(pop_labels, os.path.join(destination, f"{name}_pop_labels.csv"), index=False),
            write_table(pop_colors, os.path.join(destination, f"{name}_pop_colors.csv"), index=False),
        ]

    params = pd.DataFrame([
        {"parameter": "align_to_SS_1d", "value": ALIGN_TO[1]},
        {"parameter": "align_to_SS_2d", "value": ALIGN_TO[2]},
        {"parameter": "deme_size_N", "value": 100},
    ])
    written.append(write_table(params, os.path.join(destination, "params.csv"), index=False))
    report(written)


if __name__ == "__main__":
    main()
