#!/usr/bin/env python3
import argparse
import pandas as pd
import sys
sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")
import os

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
    parser.add_argument("--pop_labels", type=str, required=False, help="Path to population labels csv file (with columns sample_id and population and potentially sample time)")
    
    args = parser.parse_args()
    input = args.input
    output_path = args.output_path
    print(f"computing embeddings for {input}")
    l1_penalties = args.l1_penalties
    l2_penalties = args.l2_penalties
    pop_labels_df = pd.read_csv(args.pop_labels) if args.pop_labels is not None else pd.DataFrame()

    # Parsing l1 and l2 penalties
    if l1_penalties == "None":
        l1_penalties = None
    if l2_penalties == "None":
        l2_penalties = None

    if l1_penalties is not None:
        l1_penalties = {i+1: float(l1) for i, l1 in enumerate(l1_penalties.split(","))} 
    if l2_penalties is not None:
        l2_penalties = {i+1: float(l2) for i, l2 in enumerate(l2_penalties.split(","))} 

    # Load rates data
    rates_df = pd.read_parquet(input)
    input_filename = os.path.basename(input)
    se_path = os.path.join(os.path.dirname(input), "se_" + input_filename)
    se_df = pd.read_parquet(se_path) if os.path.exists(se_path) else None

    rates_df.columns = rates_df.columns.astype(float)
    if se_df is not None:
        se_df.columns = se_df.columns.astype(float)

    unique_times = sorted(rates_df.columns.tolist())
    if args.n_timepoints_emb < len(unique_times):
        idx = np.round(np.linspace(0, len(unique_times) - 1, args.n_timepoints_emb)).astype(int)
        selected_times = [unique_times[i] for i in idx]
        rates_df = rates_df[selected_times]
        if se_df is not None:
            se_df = se_df[selected_times]
        print(f"Subsampled to {len(selected_times)} timepoints "
            f"(requested {args.n_timepoints_emb} of {len(unique_times)})")

    # Parsing sample times if they exist in the population labels file, and checking if any sample times are non-zero (which would indicate ancient samples)
    if "sample_time" in pop_labels_df.columns:
        samples_in_rates = set(rates_df.index.get_level_values(0)).union(set(rates_df.index.get_level_values(1)))
        sample_times_df = pop_labels_df[["sample_id", "sample_time"]]
        sample_times_df = sample_times_df[sample_times_df['sample_id'].isin(samples_in_rates)]
        sample_times = sample_times_df.sample_time.values
        if np.any(sample_times != 0):
            print("Non-zero sample times detected:")
            print(sample_times[sample_times != 0])
        else:
            print("All sample times are zero.")
    else:
        sample_times = None

    if se_df is not None:
        print("standard errors dataframe found, will use for weighting")

    LSE = LatentSpaceEmbeddings(rates_df=rates_df, se_df=se_df, sample_times=sample_times, n_components=args.n_components, N_min=args.N_min, N_max=args.N_max, l1_strengths=l1_penalties, l2_strengths=l2_penalties)
    X_init = None

    print("Computing unsmoothed embeddings by optimizing timepoints sequentially...")
    X, loss_df = LSE.compute_embeddings_through_time(X_init=X_init)
    unsmoothed_embeddings_df = LSE.get_embeddings_df()
    unsmoothed_embeddings_df.to_parquet(f"{args.output_path}/{args.output_name}")
    loss_df.to_parquet(f"{args.output_path}/loss_{args.output_name}")

    if LSE.smoothing_params is not None:
        print("Computing smoothed embeddings...")
        # X_smooth, loss_df_smooth = LSE.compute_smooth_embeddings(X_init=X, n_epochs=args.num_epochs)
        X_smooth, loss_df_smooth = LSE.compute_embeddings_jointly(X_init=X)
        smoothed_embeddings_df = LSE.get_embeddings_df()
        smoothed_embeddings_df.to_parquet(f"{args.output_path}/smooth_{args.output_name}")
        # loss_df_smooth.to_parquet(f"{args.output_path}/loss_smooth_{args.output_name}")


    
    # if "tsinfer" in output_path or "singer" in output_path or "relate" in output_path:
    #     try:
    #         sim_embeddings_path = output_path.replace("tsinfer","sim").replace("singer","sim").replace("relate","sim") + f"/{args.output_name}"
    #         sim_embeddings_df = pd.read_parquet(sim_embeddings_path)
    #         X_init = sim_embeddings_df[["X1","X2"]].values
    #         print(f"initializing with sim embeddings from {sim_embeddings_path}")
    #     except:
    #         X_init = None
    #         print(f"could not find sim embeddings at {sim_embeddings_path}, initializing with random embeddings")
    # else:
    
    # X, loss = LSE.compute_embeddings_through_time(
    #     compute_uncertainty=args.compute_uncertainty,
    #     n_observations=args.n_observations,
    # )
    
    
    # if ancient samples are detected (samples where the embedding is NaN), print who they are
    # ancient_samples = unsmoothed_embeddings_df[unsmoothed_embeddings_df.isna().any(axis=1)]
    # if not ancient_samples.empty:
    #     print("Ancient samples detected:")
    #     # print unique sample ids
    #     print(ancient_samples.index.get_level_values(0).unique())
    # else:
    #     print("No ancient samples detected.")
    
    
if __name__ == "__main__":
    main()