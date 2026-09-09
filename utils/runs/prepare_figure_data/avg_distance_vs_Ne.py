"""
Figure data: average embedded distance against Ne, for several link functions.

Embeds a panmictic population at a range of Ne, under both link families and a
range of p, and records the average pairwise distance, the radius of the circle
fitted to the embedding's convex hull, and the link's own inverse of 1/Ne (the
distance the link maps that rate to, i.e. what a perfect embedding would give).

Note: the exponential link raises a non-finite loss inside
src/OptimizationFunctions.optimize_timepoint for the larger p (p=2 and p=4) over
part of the Ne range. Those points are recorded with a null distance rather than
aborting the whole sweep, so the rest of the figure still has its data; the
plotting script simply draws no point there. This is a pre-existing limitation of
the optimizer, not of this script.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_src_to_path, out_dir, report, write_table

add_src_to_path()
from OptimizationFunctions import optimize_timepoint, LinkFunction

FIGURE = "avg_distance_vs_Ne"

NE_LIST = np.arange(1e2, 1e4, 1e2).astype(int)
N_MIN = 1
N_MAX = 1e8
N_SAMPLES = 50
INIT_SEED = 42

LINK_FUNCTIONS = [
    {"link": "inverse_distance", "p": 0.5},
    {"link": "inverse_distance", "p": 1},
    {"link": "inverse_distance", "p": 2},
    {"link": "inverse_distance", "p": 4},
    {"link": "exponential", "p": 0.5},
    {"link": "exponential", "p": 1},
    {"link": "exponential", "p": 2},
    {"link": "exponential", "p": 4},
]


def hull_circle_radius(points):
    """Radius of the circle least-squares fitted to the convex hull of points."""
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points)
    hp = points[hull.vertices]

    x, y = hp[:, 0], hp[:, 1]
    # fit x^2 + y^2 = 2ax + 2by + c, then read the centre and radius off (a, b, c)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    rhs = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    a, b, c = sol
    return np.sqrt(c + a ** 2 + b ** 2), np.array([a, b]), hull


def compute_embeddings_panmictic(Ne, link, p, n=N_SAMPLES, N_min=N_MIN, N_max=N_MAX,
                                 init_seed=INIT_SEED):
    """Embed n samples that all coalesce with each other at rate 1/Ne."""
    np.random.seed(init_seed)
    X_init = np.random.rand(n, 2)
    R = np.ones((n, n)) * 1 / Ne
    np.fill_diagonal(R, np.nan)

    X, _, _, _ = optimize_timepoint(X_init, R, W_t=None, N_min=N_min, N_max=N_max,
                                    p=p, link=link)
    return X - np.mean(X, axis=0)   # centre the embedding


def main():
    destination = out_dir(FIGURE)

    rows = []
    failures = []
    for link_dict in LINK_FUNCTIONS:
        link, p = link_dict["link"], link_dict["p"]
        for Ne in tqdm(NE_LIST, desc=f"{link} p={p}"):
            f_inverse = LinkFunction(name=link, N_min=N_MIN, p=p).inverse(1 / Ne)
            try:
                X = compute_embeddings_panmictic(Ne, link=link, p=p)
            except FloatingPointError as exc:
                # keep the row so the sweep stays rectangular, and record why it is empty
                failures.append((link, p, int(Ne), str(exc)))
                rows.append({"link": link, "p": p, "p_label": f"{p:g}", "Ne": Ne,
                             "avg_pairwise_distance": np.nan, "hull_radius": np.nan,
                             "f_inverse": f_inverse})
                continue
            D = squareform(pdist(X, metric="euclidean"))
            rows.append({
                # p_label preserves the literal the legend prints ("1" not "1.0")
                "link": link, "p": p, "p_label": f"{p:g}", "Ne": Ne,
                "avg_pairwise_distance": np.mean(D[D != 0]),
                "hull_radius": hull_circle_radius(X)[0],
                "f_inverse": f_inverse,
            })

    if failures:
        combos = sorted({(link, p) for link, p, _, _ in failures})
        print(f"note: optimize_timepoint failed for {len(failures)} of {len(rows)} "
              f"points, in link/p combinations {combos}; their distances are null")

    params = pd.DataFrame([
        {"parameter": "N_min", "value": N_MIN},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "n_samples", "value": N_SAMPLES},
        {"parameter": "init_seed", "value": INIT_SEED},
    ])

    written = [
        write_table(pd.DataFrame(rows), os.path.join(destination, "distances.parquet")),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
