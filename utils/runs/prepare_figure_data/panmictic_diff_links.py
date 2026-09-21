"""
Figure data: panmictic embeddings under different link functions.

Embeds one panmictic population (every pair coalescing at rate 1/Ne) once per
link/p combination and saves the resulting 2D point cloud, which the figure draws
as a grid of scatter panels.

Note: the exponential link at the larger p can raise a non-finite loss inside
src/OptimizationFunctions.optimize_timepoint_newton. Such a combination is skipped with a
message rather than aborting the sweep, and the figure leaves its panel empty.
This is a pre-existing limitation of the optimizer, not of this script.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_src_to_path, out_dir, report, write_table

add_src_to_path()
from OptimizationFunctions import optimize_timepoint_newton

FIGURE = "panmictic_diff_links"

NE = 1e4
N_SAMPLES = 50
N_MIN = 1
N_MAX = 1e8
INIT_SEED = 42

LINK_FUNCTIONS = [
    {"link": "inverse_distance", "p": 0.1},
    {"link": "inverse_distance", "p": 0.5},
    {"link": "inverse_distance", "p": 1},
    {"link": "inverse_distance", "p": 2},
    {"link": "exponential", "p": 0.5},
    {"link": "exponential", "p": 1},
    {"link": "exponential", "p": 2},
    {"link": "exponential", "p": 4},
]


def compute_embeddings_panmictic(Ne, link, p, n=N_SAMPLES, N_min=N_MIN, N_max=N_MAX,
                                 init_seed=INIT_SEED):
    """Embed n samples that all coalesce with each other at rate 1/Ne."""
    np.random.seed(init_seed)
    X_init = np.random.rand(n, 2)
    R = np.ones((n, n)) * 1 / Ne
    np.fill_diagonal(R, np.nan)

    X, _, _, _ = optimize_timepoint_newton(X_init, R, W_t=None, N_min=N_min,
                                           N_max=N_max, p=p, link=link)
    return X - np.mean(X, axis=0)   # centre the embedding


def main():
    destination = out_dir(FIGURE)

    rows = []
    for order, link_dict in enumerate(LINK_FUNCTIONS):
        link, p = link_dict["link"], link_dict["p"]
        try:
            X = compute_embeddings_panmictic(NE, link, p)
        except FloatingPointError as exc:
            print(f"note: skipping link={link} p={p}: {exc}")
            continue
        rows.append(pd.DataFrame({
            # panel_order fixes the panel this combination occupies, so the figure's
            # layout does not depend on how the table happens to be sorted
            "panel_order": order,
            "link": link,
            "p": p,
            "p_label": f"{p:g}",   # the panel title prints this literal
            "sample_index": np.arange(len(X)),
            "X1": X[:, 0],
            "X2": X[:, 1],
        }))

    embeddings = pd.concat(rows, ignore_index=True)

    params = pd.DataFrame([
        {"parameter": "Ne", "value": NE},
        {"parameter": "n_samples", "value": N_SAMPLES},
        {"parameter": "N_min", "value": N_MIN},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "init_seed", "value": INIT_SEED},
    ])

    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
