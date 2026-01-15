import argparse
import pandas as pd
import sys
sys.path.append("../../../src")
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute embeddings from coalescence rates")
    parser.add_argument("--input", required=True, help="Path to rates parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for embeddings data")
    # parser.add_argument("--output_name", required=True, help="Output filename for embeddings parquet file")
    parser.add_argument("--N0", type=int, required=False, help="N0 hyperparameter for the link function")
    
    args = parser.parse_args()
    input = args.input
    rates_df = pd.read_parquet(input)
    rates = rates_df.values
    rates_df += np.min(rates[rates!=0])/100
    LSE = LatentSpaceEmbeddings(rates_df=rates_df, n_components=2)
    if args.N0:
        LSE.N0 = args.N0
        
    np.random.seed(42)
    X_init = np.random.rand(LSE.num_samples, 2)
    _ = LSE.compute_embeddings_through_time(X_init=X_init)
    embeddings_df = LSE.get_embeddings_df()
    embeddings_df.to_parquet(f"{args.output_path}/embeddings.parquet")


if __name__ == "__main__":
    main()