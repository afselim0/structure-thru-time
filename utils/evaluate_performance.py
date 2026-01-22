#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sys
sys.path.append("../../src")
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
    argparser.add_argument("--embeddings", type=str, required=True, help="Path to embeddings parquet file")
    argparser.add_argument("--rates", type=str, required=True, help="Path to true rates parquet file")
    argparser.add_argument("--colors", type=str, required=True, help="Path to population colors csv file")
    argparser.add_argument("--labels", type=str, required=True, help="Path to population labels csv file")
    argparser.add_argument("--N0", type=float, default=None, help="N0 used to compute embeddings")
    argparser.add_argument("--data_output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--plot_output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--data_output_name", type=str, required=True, help="Name of the output file")
    argparser.add_argument("--plot_output_name", type=str, required=True, help="Name of the output file")
    
    args = argparser.parse_args()

    embeddings = pd.read_parquet(args.embeddings)
    rates = pd.read_parquet(args.rates)
    pop_colors = pd.read_csv(args.colors)
    pop_labels = pd.read_csv(args.labels)

    LSE = LatentSpaceEmbeddings(rates)
    N0 = LSE.N0 if args.N0 is None else args.N0
    reconstructed_rates = reconstruct_rates(embeddings, N0=N0)
    error = performance_across_pairs(rates, reconstructed_rates)
    print(f"{args.data_output_path}/{args.data_output_name}")
    error.to_parquet(f"{args.data_output_path}/{args.data_output_name}")


    # Subsetting labels_df and colors_df to active samples only
    active_pairs = error.index.values
    active_samples = np.unique([item for sublist in active_pairs for item in sublist])
    pop_labels = pop_labels[pop_labels['sample_id'].isin(active_samples)]
    populations = pop_labels['population'].unique()
    pop_colors = pop_colors[pop_colors['population'].isin(populations)]
    if len(pop_colors) < 10:
        fig = plot_rates(error, pop_colors, pop_labels, ylabel = "Relative Error in Normalized Rates")
        fig.savefig(f"{args.plot_output_path}/{args.plot_output_name}", dpi=500)
if __name__ == "__main__":
    main()
# %%
import matplotlib.pyplot as plt
debug = False
if debug:
    rates = pd.read_parquet("../results/simulations/outputs/zigzag/data/rates_sim.V1.NPP10.NC100.FP1.WS100.T1_68266_1000.parquet")
    embeddings = pd.read_parquet("../results/simulations/outputs/zigzag/data/embeddings_sim.V1.NPP10.NC100.FP1.WS100.T1_68266_1000.N01.parquet")
    N0 = 1

    reconstructed_rates = reconstruct_rates(embeddings, N0=N0)
    # t_idx = int(4000/4)
    # plt.scatter(rates.iloc[:,t_idx], reconstructed_rates.iloc[:,t_idx])
    # plt.title(f"Time {rates.columns[t_idx]}")
    # plt.plot([0,np.max(rates.iloc[:,t_idx])], [0,np.max(rates.iloc[:,t_idx])], color="red", linestyle="--")
    plt.plot(rates.iloc[10,:],alpha=0.5, label="True Rates")
    plt.plot(reconstructed_rates.iloc[10,:],alpha=0.5, label="Reconstructed Rates")
    plt.xscale("log")

# %%
