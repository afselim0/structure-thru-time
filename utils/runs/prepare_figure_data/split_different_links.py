"""
Figure data: split-model embeddings under different link functions and N_max.

Builds a block-diagonal rate matrix standing in for a set of fully separated
populations of differing size -- every pair within a population coalesces at
1/Ne, every pair between them never does -- and embeds it once per link/p
combination at each of two N_max values, which is what the figure's two blocks of
panels compare.

Note: the exponential link at the larger p can raise a non-finite loss inside
src/OptimizationFunctions.optimize_timepoint_newton. Such a combination is skipped with a
message rather than aborting the sweep, and the figure leaves its panel empty.
This is a pre-existing limitation of the optimizer, not of this script.
"""
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_src_to_path, out_dir, report, write_table

add_src_to_path()
from OptimizationFunctions import optimize_timepoint_newton

FIGURE = "split_different_links"

NE_LIST = np.linspace(1e3, 1e4, 10).astype(int)
N_PER_POP = 20
N_MIN = 1
N_MAX_VALUES = [1e5, 1e6]
INIT_SEED = 42

LINK_FUNCTIONS = [
    {"link": "inverse_distance", "p": 0.5},
    {"link": "inverse_distance", "p": 1},
    {"link": "inverse_distance", "p": 2},
    {"link": "inverse_distance", "p": 4},
    {"link": "exponential", "p": 0.5},
    {"link": "exponential", "p": 1},
    {"link": "exponential", "p": 1.5},
    {"link": "exponential", "p": 2},
]


def construct_R(Ne_list, n=N_PER_POP):
    """Block-diagonal rate matrix: 1/Ne within each population, 0 between."""
    R_diag = np.diag(1 / np.array(Ne_list))
    R = np.kron(R_diag, np.ones((n, n)))
    np.fill_diagonal(R, np.nan)
    return R


def compute_embeddings(R, link, p, n_points, N_min=N_MIN, N_max=1e5,
                       init_seed=INIT_SEED):
    np.random.seed(init_seed)
    X_init = np.random.rand(n_points, 2)
    X, _, _, _ = optimize_timepoint_newton(X_init, R, W_t=None, N_min=N_min,
                                           N_max=N_max, p=p, link=link)
    return X - np.mean(X, axis=0)   # centre the embedding


def main():
    destination = out_dir(FIGURE)

    R = construct_R(NE_LIST, n=N_PER_POP)
    n_points = len(NE_LIST) * N_PER_POP
    # each embedded point belongs to the population whose Ne generated its block
    point_Ne = np.repeat(NE_LIST, N_PER_POP)
    point_pop = np.repeat(np.arange(len(NE_LIST)), N_PER_POP)

    rows = []
    for order, link_dict in enumerate(tqdm(LINK_FUNCTIONS)):
        link, p = link_dict["link"], link_dict["p"]
        for N_max in N_MAX_VALUES:
            try:
                X = compute_embeddings(R, link=link, p=p, n_points=n_points, N_max=N_max)
            except FloatingPointError as exc:
                print(f"note: skipping link={link} p={p} N_max={N_max:g}: {exc}")
                continue
            rows.append(pd.DataFrame({
                # panel_order fixes which panel this combination occupies, so the
                # layout does not depend on the table's sort order
                "panel_order": order,
                "link": link,
                "p": p,
                "p_label": f"{p:g}",     # the panel title prints this literal
                "N_max": N_max,
                "pop_index": point_pop,
                "Ne": point_Ne,
                "X1": X[:, 0],
                "X2": X[:, 1],
            }))

    embeddings = pd.concat(rows, ignore_index=True)

    # the Ne each colour in the figure's legend stands for, in panel order
    populations = pd.DataFrame({
        "pop_index": np.arange(len(NE_LIST)),
        "Ne": NE_LIST,
    })

    params = pd.DataFrame([
        {"parameter": "n_per_pop", "value": N_PER_POP},
        {"parameter": "N_min", "value": N_MIN},
        {"parameter": "init_seed", "value": INIT_SEED},
    ])

    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(populations, os.path.join(destination, "populations.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
