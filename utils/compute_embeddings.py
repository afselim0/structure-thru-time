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
    parser.add_argument("--n_components", type = int, default=2, help="Number of components for the latent space embeddings")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for coordinate descent optimization")
    parser.add_argument("--n_timepoints_emb", type=int, default=400, help="Number of timepoints to solve embeddings for (subsampled evenly from rates)")
    parser.add_argument("--sample_times", type=str, required=False, help="Path to sample times CSV file for ancient samples")
    
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

    print(f"computing embeddings for {input}")
    rates_df = pd.read_parquet(input)
    rates_df.columns = rates_df.columns.astype(float)
    unique_times = sorted(rates_df.columns.tolist())
    if args.n_timepoints_emb < len(unique_times):
        idx = np.round(np.linspace(0, len(unique_times) - 1, args.n_timepoints_emb)).astype(int)
        selected_times = [unique_times[i] for i in idx]
        rates_df = rates_df[selected_times]
        print(f"subsampled to {len(selected_times)} timepoints (requested {args.n_timepoints_emb} of {len(unique_times)})")
    if args.sample_times:
        samples_in_rates = set(rates_df.index.get_level_values(0)).union(set(rates_df.index.get_level_values(1)))
        sample_times_df = pd.read_csv(args.sample_times)
        sample_times_df = sample_times_df[sample_times_df['sample_id'].isin(samples_in_rates)]
        sample_times = sample_times_df.sample_time.values
        if np.any(sample_times != 0):
            print("Non-zero sample times detected:")
            print(sample_times[sample_times != 0])
        else:
            print("All sample times are zero.")
    else:
        sample_times = None
    LSE = LatentSpaceEmbeddings(rates_df=rates_df, n_components=args.n_components, N_min=args.N_min, N_max=args.N_max, l1_penalty=l1_penalties, l2_penalty=l2_penalties, sample_times=sample_times)
    output_path = args.output_path
    if "tsinfer" in output_path or "singer" in output_path or "relate" in output_path:
        try:
            sim_embeddings_path = output_path.replace("tsinfer","sim").replace("singer","sim").replace("relate","sim") + f"/{args.output_name}"
            sim_embeddings_df = pd.read_parquet(sim_embeddings_path)
            X_init = sim_embeddings_df[sim_embeddings_df.time == min(sim_embeddings_df.time)][["X1","X2"]].values
            print(f"initializing with sim embeddings from {sim_embeddings_path}")
        except:
            X_init = None
            print(f"could not find sim embeddings at {sim_embeddings_path}, initializing with random embeddings")
    else:
        X_init = None
    X, loss = LSE.compute_embeddings_through_time()


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