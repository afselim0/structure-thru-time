"""
Figure data: average embedded distance against N_min and against sample count.

Embeds a panmictic population -- every pair coalescing at the same rate 1/Ne -- and
records how far apart the embedded points end up, sweeping the minimum resolvable
population size N_min and the number of samples n in turn. Both sweeps are run at
several Ne so the figure can show how the relationship scales.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import add_src_to_path, out_dir, report, write_table

add_src_to_path()
from OptimizationFunctions import optimize_timepoint

FIGURE = "avg_distance_vs_Nmin_n"

NE_LIST = [1000, 5000, 10000, 20000]
N_MIN_LIST = np.linspace(1, 21000, 100).astype(int)
N_LIST = np.logspace(1, np.log10(500), 50).astype(int)
N_MAX = 1e8
LINK = "inverse_distance"
P = 1
# the optimizer starts from a random layout; the seed is fixed so the sweep is
# reproducible and every point is comparable to its neighbours
INIT_SEED = 42


def compute_embeddings_panmictic(Ne, link, p, n=50, N_min=1, N_max=1e8, init_seed=INIT_SEED):
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

    n_min_rows, n_rows = [], []
    for Ne in NE_LIST:
        # sweep N_min at a fixed sample count
        for N_min in tqdm(N_MIN_LIST, desc=f"Ne={Ne} N_min"):
            X = compute_embeddings_panmictic(Ne, link=LINK, p=P, n=50,
                                             N_min=N_min, N_max=N_MAX)
            n_min_rows.append({"Ne": Ne, "N_min": N_min,
                               "avg_pairwise_distance": np.nanmean(pdist(X))})
        # sweep the sample count at the smallest N_min
        for n in tqdm(N_LIST, desc=f"Ne={Ne} n"):
            X = compute_embeddings_panmictic(Ne, link=LINK, p=P, n=n,
                                             N_min=1, N_max=N_MAX)
            n_rows.append({"Ne": Ne, "n": n,
                           "avg_pairwise_distance": np.nanmean(pdist(X))})

    params = pd.DataFrame([
        {"parameter": "link", "value": LINK},
        {"parameter": "p", "value": P},
        {"parameter": "N_max", "value": N_MAX},
        {"parameter": "n_for_Nmin_sweep", "value": 50},
        {"parameter": "N_min_for_n_sweep", "value": 1},
        {"parameter": "init_seed", "value": INIT_SEED},
    ])

    written = [
        write_table(pd.DataFrame(n_min_rows),
                    os.path.join(destination, "vs_N_min.parquet")),
        write_table(pd.DataFrame(n_rows),
                    os.path.join(destination, "vs_n.parquet")),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
