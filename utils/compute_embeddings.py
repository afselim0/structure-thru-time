#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
sys.path.append("../src")
sys.path.append("../../src")
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute embeddings from coalescence rates")
    parser.add_argument("--input", required=True, help="Path to rates parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for embeddings data")
    parser.add_argument("--output_name", required=True, help="Output filename for embeddings parquet file")
    parser.add_argument("--N0", type=int, required=False, help="N0 hyperparameter for the link function")
    parser.add_argument("--sample_times", type=str, required=False, help="Path to sample times parquet file")
    
    args = parser.parse_args()
    input = args.input
    rates_df = pd.read_parquet(input)
    rates_df.columns = rates_df.columns.astype(float)
    rates = rates_df.values
    rates_df += np.min(rates[rates!=0])/100
    samples_in_rates = set(rates_df.index.get_level_values(0)).union(set(rates_df.index.get_level_values(1)))
    if args.sample_times:
        sample_times_df = pd.read_csv(args.sample_times)
        # filter to the samples in rates_df
        sample_times_df = sample_times_df[sample_times_df['sample_id'].isin(samples_in_rates)]
        sample_times = sample_times_df.sample_time.values
        # print any sample times that are not zero
        if np.any(sample_times != 0):
            print("Non-zero sample times detected:")
            print(sample_times[sample_times != 0])
        else:
            print("All sample times are zero.") 
    else:
        sample_times = None
    LSE = LatentSpaceEmbeddings(rates_df=rates_df, n_components=2, sample_times=sample_times)
    if args.N0:
        LSE.N0 = args.N0

    np.random.seed(42)
    X_init = np.random.rand(LSE.num_samples, 2)
    _ = LSE.compute_embeddings_through_time(X_init=X_init)
    embeddings_df = LSE.get_embeddings_df()
    embeddings_df.columns = embeddings_df.columns.astype(str)
    # if ancient samples are detected (samples where the embedding is NaN), print who they are
    ancient_samples = embeddings_df[embeddings_df.isna().any(axis=1)]
    if not ancient_samples.empty:
        print("Ancient samples detected:")
        # print unique sample ids
        print(ancient_samples.index.get_level_values(0).unique())
    else:
        print("No ancient samples detected.")
    embeddings_df.to_parquet(f"{args.output_path}/{args.output_name}")


if __name__ == "__main__":
    main()