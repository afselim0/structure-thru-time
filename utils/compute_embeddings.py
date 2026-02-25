#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from LatentSpaceEmbeddings import LatentSpaceEmbeddings
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute embeddings from coalescence rates")
    parser.add_argument("--input", required=True, help="Path to rates parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for embeddings data")
    parser.add_argument("--output_name", required=True, help="Output filename for embeddings parquet file")
    parser.add_argument("--N_min", type=int, required=False, help="N_min hyperparameter for the link function")
    parser.add_argument("--N_max", type=int, required=False, help="N_max hyperparameter for the link function")
    parser.add_argument("--l1_penalties", type = str, default="None", help="List of l1 penalties to compute embeddings for")
    parser.add_argument("--l2_penalties", type = str, default="None", help="List of l2 penalties to compute embeddings for")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations for optimization")
    parser.add_argument("--tol", type=float, default=1e-12, help="Tolerance for convergence in optimization")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for coordinate descent optimization")
    
    args = parser.parse_args()
    input = args.input
    l1_penalties = args.l1_penalties
    l2_penalties = args.l2_penalties

    if l1_penalties == "None":
        l1_penalties = None
    if l2_penalties == "None":
        l2_penalties = None

    if l1_penalties is not None:
        l1_penalties = {i+1: float(l1) for i, l1 in enumerate(l1_penalties.split(","))} 
    if l2_penalties is not None:
        l2_penalties = {i+1: float(l2) for i, l2 in enumerate(l2_penalties.split(","))} 

    rates_df = pd.read_parquet(input)
    LSE = LatentSpaceEmbeddings(rates_df=rates_df, n_components=2, N_min=args.N_min, N_max=args.N_max, l1_penalty=l1_penalties, l2_penalty=l2_penalties)
    if l1_penalties is None and l2_penalties is None:
        X,loss = LSE.compute_embeddings_through_time(max_iter = args.max_iter, tol = args.tol)
    else:
        X_unpenalized,loss_unpenalized = LSE.compute_embeddings_through_time(max_iter = args.max_iter, tol = args.tol)
        X,loss = LSE.compute_embeddings_coordinate_descent(X_init=X_unpenalized, max_iter = args.max_iter, tol = args.tol, num_epochs=args.num_epochs)

    embeddings_df = LSE.get_embeddings_df()
    embeddings_df.columns = embeddings_df.columns.astype(str)
    embeddings_df.to_parquet(f"{args.output_path}/{args.output_name}")


if __name__ == "__main__":
    main()