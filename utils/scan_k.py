#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from LatentSpaceEmbeddings import *
from optimization_functions import *

N_TIMEPOINTS_EMB = 100
N_MIN = 100
N_MAX = 1e7


# --- Compute embeddings for a single k value (one parallel task) ---
def _compute_for_k(k, LSE, output_path):
    """Compute and save embeddings for one k, returning its loss.

    The shared LatentSpaceEmbeddings object (and its R/W arrays) is built once
    in the parent and pickled to each worker, so only the per-k optimization
    runs in parallel. Each worker operates on its own copy, so the in-place
    mutation of self.embeddings is safe across processes.
    """
    print(f"Computing unsmoothed embeddings with k={k}...")
    X, _ = LSE.compute_embeddings_through_time(
        k=k, link="inverse_distance", compute_uncertainty=False, exact_hessian=False
    )
    embeddings_df = LSE.get_embeddings_df(id_label="sample_id")
    loss = compute_total_unpenalized_loss(
        X, LSE.R, W=LSE.W, N_min=LSE.N_min, N_max=LSE.N_max, k=k, link="inverse_distance"
    )
    embeddings_df.to_parquet(os.path.join(output_path, f"embeddings.k{k:.4f}.parquet"))
    return {"k": k, "loss": loss}


# --- Compute embeddings over a grid of k values ---
def scan_k(rates_df, output_path, se_df=None, min_k=0.8, max_k=2.0, num_k=10, n_jobs=None):
    """Compute embeddings for a grid of k values and save each as a parquet.

    Args:
        rates_df (pd.DataFrame): inferred pairwise coalescent rates.
        output_path (str): directory to write the per-k embeddings and losses.
        se_df (pd.DataFrame, optional): standard errors of the rates. Defaults to None.
        min_k (float): smallest k value in the grid.
        max_k (float): largest k value in the grid.
        num_k (int): number of k values in the grid.
        n_jobs (int, optional): number of parallel workers. Defaults to min(num_k, cpu_count).
    """
    os.makedirs(output_path, exist_ok=True)

    rates_df.columns = rates_df.columns.astype(float)

    # --- Optionally subsample the timepoints used for the embeddings ---
    unique_times = rates_df.columns
    if N_TIMEPOINTS_EMB < len(unique_times):
        idx = np.round(np.linspace(0, len(unique_times) - 1, N_TIMEPOINTS_EMB)).astype(int)
        selected_times = [unique_times[i] for i in idx]
        rates_df = rates_df[selected_times]
        if se_df is not None:
            se_df.columns = se_df.columns.astype(float)
            se_df = se_df[selected_times]
        print(f"Subsampled to {len(selected_times)} timepoints "
              f"(requested {N_TIMEPOINTS_EMB} of {len(unique_times)})")

    # --- Build the shared object once (R/W construction is k-independent) ---
    LSE = LatentSpaceEmbeddings(rates_df, se_df=se_df, N_min=N_MIN, N_max=N_MAX)
    k_values = np.linspace(min_k, max_k, num_k)

    # --- Parallelize the per-k optimization over k ---
    if n_jobs is None:
        n_jobs = min(len(k_values), os.cpu_count())
    print(f"Scanning {len(k_values)} k values across {n_jobs} workers...")
    with ProcessingPool(nodes=n_jobs) as pool:
        loss_records = pool.map(
            _compute_for_k,
            k_values,
            [LSE] * len(k_values),
            [output_path] * len(k_values),
        )

    # --- Collect and save the losses across k ---
    losses_df = pd.DataFrame(loss_records).sort_values("k").reset_index(drop=True)
    losses_df.to_csv(os.path.join(output_path, "losses.csv"), index=False)
    print(f"Saved {len(k_values)} embeddings and losses to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan a grid of k values, saving per-k embeddings and losses."
    )
    parser.add_argument("--rates_df", required=True, help="Path to the rates parquet file.")
    parser.add_argument("--output_path", required=True, help="Directory to write outputs.")
    parser.add_argument("--se_df", default=None, help="Path to the standard-errors parquet file.")
    parser.add_argument("--min_k", type=float, default=1.0, help="Smallest k value.")
    parser.add_argument("--max_k", type=float, default=2.0, help="Largest k value.")
    parser.add_argument("--num_k", type=int, default=10, help="Number of k values.")
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Number of parallel workers. Defaults to min(num_k, cpu_count).")
    args = parser.parse_args()

    rates_df = pd.read_parquet(args.rates_df)
    se_df = pd.read_parquet(args.se_df) if args.se_df is not None else None

    scan_k(
        rates_df=rates_df,
        output_path=args.output_path,
        se_df=se_df,
        min_k=args.min_k,
        max_k=args.max_k,
        num_k=args.num_k,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
#%%
import matplotlib.pyplot as plt
import pandas as pd
from plotting_style import *
losses = pd.read_csv("../results/empirical/Wohns/all.npp.2/diff_k/losses.csv")
losses["likelihood"] = -losses["loss"]
fig, ax = plt.subplots(dpi=500)
ax.plot(losses["k"], losses["likelihood"], marker="o")
ax.set_xlabel("k")
ax.set_ylabel("Log-Likelihood")
ax.set_title("Wohns 2 samples per population - Log-Likelihood vs k")
fig.savefig("../results/empirical/Wohns/all.npp.2/diff_k/loglikelihood_vs_k.png", bbox_inches="tight")

# %%
# --- Fast scan: compute only r2 / log_r2 / Spearman vs k, skipping the scatter rendering ---
from scipy.stats import spearmanr
from evaluate_performance import *
import numpy as np
from tqdm import tqdm
input_rates = pd.read_parquet("../results/empirical/Wohns/all.npp.2/data/rates.minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet")
pop_labels = pd.read_csv("../data/metadata/wohns_pop_labels.csv")
pop_colors = pd.read_csv("../data/metadata/wohns_pop_colors.csv")
losses = pd.read_csv("../results/empirical/Wohns/all.npp.2/diff_k/losses.csv")
k_values = losses["k"].values

# Match the input-rate preprocessing used inside reconstructed_vs_input_all_times
input_x = input_rates.copy()
input_x += 1 / N_MAX
input_x.columns = input_x.columns.astype(float)
k_values = losses["k"].values
metrics_records = []
for k in tqdm(k_values, total=len(k_values)):
    embeddings = pd.read_parquet(f"../results/empirical/Wohns/all.npp.2/diff_k/embeddings.k{k:.4f}.parquet")
    embeddings.rename(columns={"individual_id": "sample_id"}, inplace=True)
    D_df = compute_distance_df(embeddings, id_label="sample_id")
    reconstructed_rates = reconstruct_rates(D_df, N_MIN, k=k, link="inverse_distance")
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    # Align on common pairs (index) and timepoints (columns) before comparing —
    # reconstructed rates cover only the subsampled times and embedded pairs
    common_idx = input_x.index.intersection(reconstructed_rates.index)
    common_cols = input_x.columns.intersection(reconstructed_rates.columns)
    x_df = input_x.loc[common_idx, common_cols]
    y_df = reconstructed_rates.loc[common_idx, common_cols]

    # Flatten and keep only finite, strictly-positive pairs (same validity mask as the full plot)
    x_flat = x_df.values.ravel()
    y_flat = y_df.values.ravel()
    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 0) & (y_flat > 0)
    x_v, y_v = x_flat[valid], y_flat[valid]

    r2 = np.corrcoef(x_v, y_v)[0, 1] ** 2
    log_r2 = np.corrcoef(np.log(x_v), np.log(y_v))[0, 1] ** 2
    spearman_corr, _ = spearmanr(x_v, y_v)
    metrics_records.append({"k": k, "r2": r2, "log_r2": log_r2, "spearman": spearman_corr})

metrics_df = pd.DataFrame(metrics_records)

# --- Plot the three goodness-of-fit metrics as a function of k ---
fig, ax = plt.subplots(dpi=500)
ax.plot(metrics_df["k"], metrics_df["r2"], marker="o", label="$r^2$")
ax.plot(metrics_df["k"], metrics_df["log_r2"], marker="s", label="$r^2$ (log)")
ax.plot(metrics_df["k"], metrics_df["spearman"], marker="^", label="Spearman $\\rho$")
ax.set_xlabel("k")
ax.set_ylabel("Goodness of fit")
ax.set_title("Wohns 2 samples per population - fit vs k")
ax.legend()
plt.show()
fig.savefig("../results/empirical/Wohns/all.npp.2/diff_k/fit_vs_k.png", bbox_inches="tight")
# %%
