#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from scipy.stats import spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
from OptimizationFunctions import (compute_distance_df, reconstruct_rates,
                                   compute_unpenalized_loss, LinkFunction)
from evaluate_performance import align_time_columns


# --- Goodness-of-fit metrics between an (input or true) rates df and reconstructed rates ---
def _compute_metrics(rates_df, reconstructed_rates_df, N_max, label):
    """Align rates onto the reconstructed-rates time grid, then compute r2, r2_log, spearman_rho.

    Mirrors the validity-masked comparison used throughout evaluate_performance.py: only
    finite, strictly-positive (rate, reconstructed_rate) pairs contribute to the metrics.
    """
    rates_df = rates_df.copy()
    rates_df += 1 / N_max
    rates_df, reconstructed_rates_df = align_time_columns(rates_df, reconstructed_rates_df, label)

    common_idx = rates_df.index.intersection(reconstructed_rates_df.index)
    x_df = rates_df.loc[common_idx]
    y_df = reconstructed_rates_df.loc[common_idx]

    x_flat = x_df.values.ravel()
    y_flat = y_df.values.ravel()
    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 0) & (y_flat > 0)
    x_v, y_v = x_flat[valid], y_flat[valid]

    r2 = np.corrcoef(x_v, y_v)[0, 1] ** 2
    r2_log = np.corrcoef(np.log(x_v), np.log(y_v))[0, 1] ** 2
    spearman_rho, _ = spearmanr(x_v, y_v)
    return r2, r2_log, spearman_rho


# --- Compute embeddings for a single p value (one parallel task) ---
def _compute_for_p(p, LSE, output_path, link, id_label, input_rates_df, true_rates_df, N_max):
    """Fit embeddings at a single p, save them, and return loss + goodness-of-fit records.

    The shared LatentSpaceEmbeddings object (and its R/W arrays) is built once in the
    parent and pickled to each worker, so only the per-p optimization runs in parallel.
    Each worker operates on its own copy, so the in-place mutation of self.embeddings
    is safe across processes.
    """
    print(f"Computing embeddings with p={p}...")
    # Exact-Hessian covariances are only implemented for p == 1; this scan sweeps p away from 1
    X, _ = LSE.compute_embeddings_through_time(
        p=p, link=link, compute_uncertainty=False, exact_hessian=False
    )
    embeddings_df = LSE.get_embeddings_df(id_label=id_label)
    loss = compute_unpenalized_loss(
        X, LSE.R.astype(float) + 1.0 / LSE.N_max, W=LSE.W.astype(float),
        link=LinkFunction(name=link, N_min=LSE.N_min, p=p)
    )
    embeddings_df.to_parquet(os.path.join(output_path, f"embeddings.p{p:.4g}.parquet"))

    D_df = compute_distance_df(embeddings_df, id_label=id_label)
    reconstructed_rates_df = reconstruct_rates(D_df, LSE.N_min, p=p, link=link)
    reconstructed_rates_df.columns = reconstructed_rates_df.columns.astype(float)
    reconstructed_rates_df.to_parquet(os.path.join(output_path, f"reconstructed_rates.p{p:.4g}.parquet"))

    r2, r2_log, spearman_rho = _compute_metrics(input_rates_df, reconstructed_rates_df, N_max, label="input")
    records = [{"p": p, "loss": loss, "r2": r2, "r2_log": r2_log, "spearman_rho": spearman_rho, "vs": "input"}]

    if true_rates_df is not None:
        r2_t, r2_log_t, spearman_rho_t = _compute_metrics(true_rates_df, reconstructed_rates_df, N_max, label="true")
        records.append({"p": p, "loss": loss, "r2": r2_t, "r2_log": r2_log_t, "spearman_rho": spearman_rho_t, "vs": "true"})

    return records


# --- Compute embeddings over a grid of p values ---
def scan_p(rates_df, output_path, se_df=None, true_rates_df=None, min_p=0.5, max_p=2.0, num_p=10,
           N_min=1, N_max=1e7, link="inverse_distance", n_components=2, n_timepoints_emb=400,
           id_label="sample_id", n_jobs=None):
    """Compute embeddings for a grid of p values (fixed n_components), saving each as a parquet
    plus a single dataframe of per-p diagnostics (loss, r2, r2_log, spearman_rho).
    """
    os.makedirs(output_path, exist_ok=True)

    rates_df.columns = rates_df.columns.astype(float)
    input_rates_df = rates_df.copy()

    # --- Optionally subsample the timepoints used for the embeddings ---
    unique_times = rates_df.columns
    if n_timepoints_emb < len(unique_times):
        idx = np.round(np.linspace(0, len(unique_times) - 1, n_timepoints_emb)).astype(int)
        selected_times = [unique_times[i] for i in idx]
        rates_df = rates_df[selected_times]
        if se_df is not None:
            se_df.columns = se_df.columns.astype(float)
            se_df = se_df[selected_times]
        print(f"Subsampled to {len(selected_times)} timepoints "
              f"(requested {n_timepoints_emb} of {len(unique_times)})")

    # --- Build the shared object once (R/W construction is p-independent) ---
    LSE = LatentSpaceEmbeddings(rates_df, se_df=se_df, N_min=N_min, N_max=N_max, n_components=n_components)
    p_values = np.linspace(min_p, max_p, num_p)

    # --- Parallelize the per-p optimization over p ---
    if n_jobs is None:
        n_jobs = min(len(p_values), os.cpu_count())
    print(f"Scanning {len(p_values)} p values across {n_jobs} workers...")
    with ProcessingPool(nodes=n_jobs) as pool:
        record_lists = pool.map(
            _compute_for_p,
            p_values,
            [LSE] * len(p_values),
            [output_path] * len(p_values),
            [link] * len(p_values),
            [id_label] * len(p_values),
            [input_rates_df] * len(p_values),
            [true_rates_df] * len(p_values),
            [N_max] * len(p_values),
        )

    # --- Collect and save the diagnostics across p ---
    metrics_df = pd.DataFrame([r for records in record_lists for r in records])
    metrics_df = metrics_df.sort_values(["p", "vs"]).reset_index(drop=True)
    return metrics_df


def main():
    parser = argparse.ArgumentParser(description="Scan a grid of p values, saving per-p embeddings and diagnostics.")
    parser.add_argument("--rates", required=True, help="Path to the input (estimated) rates parquet file.")
    parser.add_argument("--output_path", required=True, help="Directory to write per-p embeddings/reconstructed rates.")
    parser.add_argument("--output", required=True, help="Path to write the diagnostics csv (p, loss, r2, r2_log, spearman_rho, vs).")
    parser.add_argument("--se_df", default=None, help="Path to the standard-errors parquet file.")
    parser.add_argument("--true_rates", default=None, help="Path to ground-truth demographic rates parquet file, if available.")
    parser.add_argument("--min_p", type=float, default=0.5, help="Smallest p value.")
    parser.add_argument("--max_p", type=float, default=2.0, help="Largest p value.")
    parser.add_argument("--num_p", type=int, default=10, help="Number of p values.")
    parser.add_argument("--N_min", type=float, required=True, help="N_min hyperparameter for the link function.")
    parser.add_argument("--N_max", type=float, required=True, help="N_max hyperparameter for the link function.")
    parser.add_argument("--link", type=str, default="inverse_distance", help="Link function type (inverse_distance or exponential).")
    parser.add_argument("--n_components", type=int, default=2, help="Number of embedding dimensions (held fixed while p is scanned).")
    parser.add_argument("--n_timepoints_emb", type=int, default=400, help="Number of timepoints to solve embeddings for (subsampled evenly from rates).")
    parser.add_argument("--pop_labels", type=str, default=None, help="Path to population labels csv file (unused directly, accepted for CLI symmetry with compute_embeddings.py).")
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true", help="If set, rates are per-haplotype (sample_id). Otherwise per-individual (individual_id).")
    parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel workers. Defaults to min(num_p, cpu_count).")
    args = parser.parse_args()

    id_label = "sample_id" if args.per_haplotype else "individual_id"
    rates_df = pd.read_parquet(args.rates)
    se_df = pd.read_parquet(args.se_df) if args.se_df is not None else None
    true_rates_df = pd.read_parquet(args.true_rates) if args.true_rates is not None else None
    if true_rates_df is not None:
        true_rates_df.columns = true_rates_df.columns.astype(float)

    metrics_df = scan_p(
        rates_df=rates_df, output_path=args.output_path, se_df=se_df, true_rates_df=true_rates_df,
        min_p=args.min_p, max_p=args.max_p, num_p=args.num_p,
        N_min=args.N_min, N_max=args.N_max, link=args.link,
        n_components=args.n_components, n_timepoints_emb=args.n_timepoints_emb,
        id_label=id_label, n_jobs=args.n_jobs,
    )
    metrics_df.to_csv(args.output, index=False)
    print(f"Saved {args.num_p} embeddings and diagnostics to {args.output_path} / {args.output}")


if __name__ == "__main__":
    main()
