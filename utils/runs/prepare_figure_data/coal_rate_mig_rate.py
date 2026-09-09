"""
Figure data: coalescence rate against migration rate.

Two tables, one per panel group of the figure:
  - rate trajectories, within and between two symmetrically migrating populations,
    at a handful of Nm values
  - the plateau each between-population trajectory saturates at, and the rate
    constant of its approach, across a finer grid of migration rates

Both come from msprime's analytic coalescence-rate trajectories rather than from
simulation, so no seed is involved.
"""
import os
import sys

import numpy as np
import pandas as pd
import msprime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import out_dir, report, write_table

FIGURE = "coal_rate_mig_rate"

Ne_A = 1e4
Ne_B = 1e4

# panel group 1: trajectories at four scaled migration rates
NM_LIST = [1e-3, 1e-2, 1e-1, 1]
TRAJECTORY_TIMES = np.arange(1, 200001, 100)

# panel group 2: saturation behaviour across a migration-rate grid
M_GRID = np.linspace(1e-7, 1e-4, 100)
SATURATION_TIMES = np.arange(1, 300001, 100)
# the approach to the plateau is fit over the first stretch of the trajectory,
# where it is still close to exponential
FIT_POINTS = 200


def two_pop_migration(Ne_A, Ne_B, m):
    demography = msprime.Demography.isolated_model([Ne_A, Ne_B])
    demography.set_migration_rate(source=0, dest=1, rate=m)
    demography.set_migration_rate(source=1, dest=0, rate=m)
    return demography


def main():
    destination = out_dir(FIGURE)

    # ---- trajectories ----
    rows = []
    for Nm in NM_LIST:
        m = Nm / Ne_A
        debugger = two_pop_migration(Ne_A, Ne_B, m).debug()
        between, _ = debugger.coalescence_rate_trajectory(
            steps=TRAJECTORY_TIMES, lineages={0: 1, 1: 1})
        within, _ = debugger.coalescence_rate_trajectory(
            steps=TRAJECTORY_TIMES, lineages={0: 2, 1: 0})
        rows.append(pd.DataFrame({
            "Nm": Nm,
            # the panel title prints this verbatim, so the literal is carried through
            # rather than reformatted from the float on the way out
            "Nm_label": f"{Nm:g}",
            "m": m,
            "time": TRAJECTORY_TIMES,
            "between_rate": between,
            "within_rate": within,
        }))
    trajectories = pd.concat(rows, ignore_index=True)

    # ---- saturation limit and rate of convergence ----
    plateaus, rates, intercepts = [], [], []
    for m in tqdm(M_GRID):
        debugger = two_pop_migration(Ne_A, Ne_B, m).debug()
        between, _ = debugger.coalescence_rate_trajectory(
            steps=SATURATION_TIMES, lineages={0: 1, 1: 1})
        # the trajectory saturates, so its last value stands in for the plateau; the
        # epsilon keeps the log below finite when the two coincide exactly
        A = between[-1] + 1e-10
        plateaus.append(A)
        y = -np.log(1 - between / A)
        slope, intercept = np.polyfit(SATURATION_TIMES[:FIT_POINTS], y[:FIT_POINTS], 1)
        rates.append(slope)
        intercepts.append(intercept)

    saturation = pd.DataFrame({
        "m": M_GRID,
        "Nm": M_GRID * Ne_A,
        "plateau": plateaus,
        "rate_constant": rates,
        "intercept": intercepts,
    })

    params = pd.DataFrame([
        {"parameter": "Ne_A", "value": Ne_A},
        {"parameter": "Ne_B", "value": Ne_B},
        {"parameter": "fit_points", "value": FIT_POINTS},
    ])

    written = [
        write_table(trajectories, os.path.join(destination, "rate_trajectories.parquet")),
        write_table(saturation, os.path.join(destination, "saturation.parquet")),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
