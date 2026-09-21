#!/usr/bin/env python3
"""Genotype PCA over a directory of tree sequences.

The genotype matrix of the whole dataset is never materialised. Each tree
sequence contributes its sites to a running genetic relatedness matrix

    G = sum_chunks Z_c^T Z_c,     Z_c = (g_c - mean) / sd

(columns mean-centred and scaled to unit variance, as requested), and the
principal components are the eigenvectors of G / n_sites. This is exact — it
is the same decomposition a single-shot PCA on the stacked matrix would give —
but it costs O(n^2) memory instead of O(sites x n).

The alternative of concatenating the chunks and calling TreeSequence.pca() is
not used, for two reasons: tskit 1.0.3's pca() only implements mode="branch"
(site mode raises TSK_ERR_UNSUPPORTED_STAT_MODE), which is a different
estimand from a genotype PCA; and these chunks each carry the full chromosome
coordinate space, so concatenating them yields a sequence length equal to
n_chunks x chromosome_length rather than a contiguous genome.
"""
import argparse
import os

# This cluster's OpenBLAS thread-thrashes on the small (n x n) products below,
# and the per-chunk work is already parallelised across processes, so pin BLAS
# to one thread before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import glob
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tskit
from tqdm import tqdm


def _chunk_grm(args):
    """Return (Z^T Z, n_sites_used) for one tree sequence.

    Monomorphic sites carry no information and would divide by a zero standard
    deviation, so they are dropped rather than scaled.
    """
    ts_path, sample_ids = args
    ts = tskit.load(ts_path)
    g = ts.genotype_matrix(samples=sample_ids).astype(np.float32)  # (sites, samples)
    if g.shape[0] == 0:
        n = len(sample_ids)
        return np.zeros((n, n)), 0

    # A site is monomorphic when it has no variance *across samples*, i.e. along
    # axis 1 of this (sites, samples) matrix. Such sites carry no information and
    # would divide by zero when standardised, so drop them first.
    site_sd = g.std(axis=1)
    polymorphic = site_sd > 0
    if not polymorphic.any():
        n = len(sample_ids)
        return np.zeros((n, n)), 0
    g = g[polymorphic]

    # Standardise each site (row) to mean 0 / unit variance across samples, so
    # every site contributes equally to the relatedness matrix.
    z = (g - g.mean(axis=1, keepdims=True)) / g.std(axis=1, keepdims=True)
    return (z.T @ z).astype(np.float64), g.shape[0]


def compute_pca(ts_dir, sample_ids, num_components=2, num_threads=1):
    """Accumulate the GRM across every tree sequence in ts_dir, then decompose it."""
    ts_paths = sorted(glob.glob(os.path.join(ts_dir, "*.trees")))
    if not ts_paths:
        raise ValueError(f"No .trees files found in {ts_dir}")
    print(f"running PCA over {len(ts_paths)} tree sequences ({len(sample_ids)} samples)")

    n = len(sample_ids)
    G = np.zeros((n, n))
    n_sites = 0
    work = [(p, sample_ids) for p in ts_paths]

    if num_threads > 1:
        with ProcessPoolExecutor(max_workers=num_threads) as pool:
            for G_c, s_c in tqdm(pool.map(_chunk_grm, work), total=len(work),
                                 desc="Accumulating GRM"):
                G += G_c
                n_sites += s_c
    else:
        for item in tqdm(work, desc="Accumulating GRM"):
            G_c, s_c = _chunk_grm(item)
            G += G_c
            n_sites += s_c

    if n_sites == 0:
        raise ValueError(f"No polymorphic sites found across the tree sequences in {ts_dir}")
    G /= n_sites

    # G is symmetric PSD, so eigh is both correct and cheaper than an SVD.
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    order = np.argsort(eigenvalues)[::-1][:num_components]
    eigenvalues = eigenvalues[order]
    factors = eigenvectors[:, order]

    total_variance = np.trace(G)
    explained = eigenvalues / total_variance if total_variance > 0 else np.full(num_components, np.nan)
    print(f"used {n_sites} polymorphic sites; "
          f"variance explained: {np.round(explained * 100, 2)}%")

    pca_df = pd.DataFrame(factors, columns=[f"PC{i + 1}" for i in range(num_components)])
    pca_df.insert(0, "sample_id", np.asarray(sample_ids, dtype=int))
    return pca_df, explained


def main():
    parser = argparse.ArgumentParser(description="Run genotype PCA on a directory of tree sequences")
    parser.add_argument("--input_path", required=True, type=str, help="Directory containing the .trees files")
    parser.add_argument("--output_path", required=True, type=str, help="Output directory for the PCA csv")
    parser.add_argument("--output_name", required=True, type=str, help="Output csv filename")
    parser.add_argument("--pop_labels", required=True, type=str,
                        help="Path to population labels csv (already sliced to the samples used in this run)")
    parser.add_argument("--num_components", type=int, default=2, help="Number of principal components to retain")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of worker processes")

    args = parser.parse_args()
    pop_labels = pd.read_csv(args.pop_labels)
    sample_ids = pop_labels["sample_id"].values

    os.makedirs(args.output_path, exist_ok=True)
    pca_df, explained = compute_pca(
        ts_dir=args.input_path,
        sample_ids=sample_ids,
        num_components=args.num_components,
        num_threads=args.num_threads,
    )

    # The plotting rule reads the explained variance back off these columns for
    # its axis labels, so they travel with the coordinates in the same csv.
    for i, frac in enumerate(explained):
        pca_df[f"PC{i + 1}_explained"] = frac

    out = os.path.join(args.output_path, args.output_name)
    pca_df.to_csv(out, index=False)
    print(f"saved PCA coordinates to {out}")


if __name__ == "__main__":
    main()
