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
    
    args = parser.parse_args()
    input = args.input
    rates_df = pd.read_parquet(input)
    rates_df.columns = rates_df.columns.astype(float)
    rates = rates_df.values
    rates_df += np.min(rates[rates!=0])/100
    LSE = LatentSpaceEmbeddings(rates_df=rates_df, n_components=2, N_min=args.N_min, N_max=args.N_max)
    np.random.seed(42)
    X_init = np.random.rand(LSE.num_samples, 2)
    _ = LSE.compute_embeddings_through_time(X_init=X_init)
    embeddings_df = LSE.get_embeddings_df()
    embeddings_df.columns = embeddings_df.columns.astype(str)
    embeddings_df.to_parquet(f"{args.output_path}/{args.output_name}")


if __name__ == "__main__":
    main()