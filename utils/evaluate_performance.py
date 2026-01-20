#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
sys.path.append("../src")
from LatentSpaceEmbeddings import LatentSpaceEmbeddings, compute_f
from plot_rates import plot_rates
import argparse

def reconstruct_rates(embeddings, N0):
    embeddings_pairs = embeddings.merge(embeddings, on="time", suffixes=("_1", "_2"))
    embeddings_pairs = embeddings_pairs[embeddings_pairs["sample_id_1"] < embeddings_pairs["sample_id_2"]]
    embeddings_pairs["distance"] = np.sqrt(
    (embeddings_pairs["X1_1"] - embeddings_pairs["X1_2"])**2 +
    (embeddings_pairs["X2_1"] - embeddings_pairs["X2_2"])**2)

    D_df = (embeddings_pairs
    .set_index(["sample_id_1", "sample_id_2", "time"])["distance"]
    .unstack("time"))

    reconstructed_rates,_ = compute_f(D_df, N0)
    return reconstructed_rates

def performance_across_pairs(rates, reconstructed_rates):
    rates_norm = rates.div(rates.sum(axis=1), axis=0)
    reconstructed_rates_norm = reconstructed_rates.div(reconstructed_rates.sum(axis=1), axis=0)

    error = np.abs(rates_norm - reconstructed_rates_norm)
    error /= np.mean(rates_norm)
    return error

def main():
    argparser = argparse.ArgumentParser(description="Evaluate performance of rate reconstruction from embeddings")
    argparser.add_argument("--embeddings_path", type=str, required=True, help="Path to embeddings parquet file")
    argparser.add_argument("--rates_path", type=str, required=True, help="Path to true rates parquet file")
    argparser.add_argument("--pop_colors_path", type=str, required=True, help="Path to population colors csv file")
    argparser.add_argument("--pop_labels_path", type=str, required=True, help="Path to population labels csv file")
    argparser.add_argument("--N0", type=float, default=None, help="N0 used to compute embeddings")
    argparser.add_argument("--output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--output_name", type=str, required=True, help="Name of the output file")
    args = argparser.parse_args()

    embeddings = pd.read_parquet(args.embeddings_path)
    rates = pd.read_parquet(args.rates_path)
    pop_colors = pd.read_csv(args.pop_colors_path)
    pop_labels = pd.read_csv(args.pop_labels_path)

    LSE = LatentSpaceEmbeddings(rates)
    N0 = LSE.N0 if args.N0 is None else args.N0
    reconstructed_rates = reconstruct_rates(embeddings, N0=N0)
    error = performance_across_pairs(rates, reconstructed_rates)
    fig = plot_rates(error, pop_colors, pop_labels, ylabel = "Relative Error in Normalized Rates", title="Reconstruction Error of Pairwise Rates")

    error.to_parquet(f"{args.output_path}/{args.output_name}.parquet")
    fig.savefig(f"{args.output_path}/{args.output_name}.png", dpi=500)

if __name__ == "__main__":
    main()