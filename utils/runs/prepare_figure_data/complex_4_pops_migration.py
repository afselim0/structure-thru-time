"""
Figure data: the complex four-population example under increasing migration.

Panels A-C are the embeddings from the three migration runs, each Procrustes-aligned
to the no-migration run so that the panels differ by the demography rather than by
the arbitrary rotation/reflection the optimizer happened to settle on. The
alignment is data work, so it is done here and the aligned coordinates are what
the figure reads.

Panel D contrasts each population's pair coalescence rate against the 1/(2Ne)
baseline it would have with no migration; both come from msprime's analytic
trajectories for the demes files, computed here rather than in the figure.
"""
import os
import sys

import numpy as np
import pandas as pd
import demes
import msprime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (DEMES_DIR, OUTPUTS_DIR, add_src_to_path, find_output_file,
                     out_dir, report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "complex_4_pops_migration"

D, N_MIN, N_MAX, P = 2, 1000, 1000000, 1

# the reference run (no migration) and the three migration runs aligned to it.
# each run's outputs live under complex_4_pops/<run name>/sim/data
BASE_DIR = os.path.join(OUTPUTS_DIR, "simulations", "complex_4_pops")
RUNS = {
    "0": ("complex_4_pops_m_0", "complex_4_pops.yaml"),
    "0.01": ("complex_4_pops_m_01", "complex_4_pops_m_01.yaml"),
    "0.001": ("complex_4_pops_m_001", "complex_4_pops_m_001.yaml"),
    "0.0001": ("complex_4_pops_m_0001", "complex_4_pops_m_0001.yaml"),
}
REFERENCE_M = "0"

# panel D's time axis, and the two populations it follows
RECENT_TIME = np.arange(0, 1000)
RATE_POPULATIONS = ["A", "B"]


def run_data_dir(run_name):
    """The data directory of one migration run."""
    return os.path.join(BASE_DIR, run_name, "sim", "data")


def main():
    destination = out_dir(FIGURE)

    # ---- embeddings, aligned to the no-migration run ----
    embeddings = {}
    for m, (run_name, _) in RUNS.items():
        path = find_output_file(run_data_dir(run_name), "embeddings",
                                d=D, Nmin=N_MIN, Nmax=N_MAX, p=P)
        embeddings[m] = pd.read_parquet(path)

    reference = embeddings[REFERENCE_M]
    aligned = []
    for m, df in embeddings.items():
        if m != REFERENCE_M:
            df = procrustes_align_trajectories_df(reference, df, allow_reflections=True)
        aligned.append(df.assign(m=m))
    aligned_embeddings = pd.concat(aligned, ignore_index=True)

    # ---- panel D: analytic coalescence rates and the no-migration 1/(2Ne) baseline ----
    rate_rows = []
    for m, (_, demes_file) in RUNS.items():
        debugger = msprime.Demography.from_demes(
            demes.load(require(os.path.join(DEMES_DIR, demes_file)))).debug()
        for pop in RATE_POPULATIONS:
            rates = debugger.coalescence_rate_trajectory(
                RECENT_TIME, lineages={pop: 2})[0]
            rate_rows.append(pd.DataFrame({
                "m": m, "population": pop, "time": RECENT_TIME, "rate": rates}))
    rates_table = pd.concat(rate_rows, ignore_index=True)

    # the baseline is 1/(2Ne) under the no-migration demography: the rate a pair
    # would coalesce at with no migration at all
    baseline_debugger = msprime.Demography.from_demes(
        demes.load(os.path.join(DEMES_DIR, RUNS[REFERENCE_M][1]))).debug()
    Ne = baseline_debugger.population_size_trajectory(steps=RECENT_TIME)
    populations = [p.name for p in baseline_debugger.demography.populations]
    baseline_rows = []
    for pop in RATE_POPULATIONS:
        column = populations.index(pop)
        baseline_rows.append(pd.DataFrame({
            "population": pop, "time": RECENT_TIME,
            "inverse_2Ne": 1 / (2 * Ne[:, column])}))
    baseline = pd.concat(baseline_rows, ignore_index=True)

    pop_labels = pd.read_csv(require(os.path.join(run_data_dir(RUNS[REFERENCE_M][0]), "pop_labels.csv")))
    pop_colors = pd.read_csv(require(os.path.join(run_data_dir(RUNS[REFERENCE_M][0]), "pop_colors.csv")))
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
    pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]

    written = [
        write_table(aligned_embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(rates_table, os.path.join(destination, "coalescence_rates.parquet")),
        write_table(baseline, os.path.join(destination, "inverse_2Ne.parquet")),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
