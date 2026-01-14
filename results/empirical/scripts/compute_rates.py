import argparse
import os
import sys
sys.path.append("../../../src")
from CountCoalEvents import CountCoalEvents
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--input_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for rates data")
    parser.add_argument("--max_time", type=int, required=True, help="Maximum time for rate calculation")
    parser.add_argument("--window_size", type=int, required=True, help="Window size for rate calculation")
    parser.add_argument("--n_timepoints", type=int, required=True, help="Number of timepoints for rate calculation")
    parser.add_argument("--num_posterior_samples", type=int, default=1, help="Number of posterior samples to use")
    parser.add_argument("--fraction_pairs", type=float, default=1, help="Fraction of pairs to sample for coalescence counting")
    parser.add_argument("--pops", nargs='+', help="Population labels to compute rates for")
    parser.add_argument("--n_samples_per_pop", type=int, help="Number of samples to draw per population")
    parser.add_argument("--pop_labels", type=str, help="Path to population labels file")
    
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    max_time = args.max_time
    window_size = args.window_size
    n_timepoints = args.n_timepoints
    num_posterior_samples = args.num_posterior_samples
    fraction_pairs = args.fraction_pairs
    chosen_pops = args.pops
    n_samples_per_pop = args.n_samples_per_pop
    pop_labels_path = args.pop_labels
    population_labels = pd.read_csv(pop_labels_path)

    os.makedirs(output_path, exist_ok=True)

    if chosen_pops is None:
        population_labels = population_labels[population_labels["sample_time"]==0]
    else:
        population_labels = population_labels[population_labels["population"].isin(chosen_pops)]
    if n_samples_per_pop is not None:
        chosen_samples = []
        for pop in chosen_pops:
            sample_ids = population_labels[population_labels["population"] == pop]["sample_id"].values
            if n_samples_per_pop < len(sample_ids):
                subsample_ids = sample_ids[:n_samples_per_pop]
            else:
                subsample_ids = sample_ids
            chosen_samples.extend(subsample_ids)

    data_dir = os.path.join(output_path, "rates_data")
    os.makedirs(data_dir, exist_ok=True)

    CCE = CountCoalEvents(ts_dir=input_path, num_posterior_samples=num_posterior_samples,max_time=max_time, num_timepoints=n_timepoints,fraction_pairs=fraction_pairs, sample_ids=chosen_samples)
    pdf_df = CCE.compute_coal_time_pdf(num_threads=os.cpu_count()-1, data_output=data_dir)
    rates_df = CCE.compute_coal_rates(pdf_df=pdf_df, window_size=window_size)
    rates_df.to_parquet(f"{output_path}/rates.parquet")

if __name__ == "__main__":
    main()

#%%
input_path = "../../../data/trees/wohns_chopped"
output_path = "../wohns/test"
max_time = 2000
window_size = 100
n_timepoints = 2000
num_posterior_samples = 1
fraction_pairs = 1
chosen_pops = ["YRI","CEU"]
n_samples_per_pop = 10
population_labels = pd.read_csv("wohns_pop_labels.csv")
# %%
if chosen_pops is None:
    population_labels = population_labels[population_labels["sample_time"]==0]
else:
    population_labels = population_labels[population_labels["population"].isin(chosen_pops)]
if n_samples_per_pop is not None:
    chosen_samples = []
    for pop in chosen_pops:
        sample_ids = population_labels[population_labels["population"] == pop]["sample_id"].values
        if n_samples_per_pop < len(sample_ids):
            subsample_ids = sample_ids[:n_samples_per_pop]
        else:
            subsample_ids = sample_ids
        chosen_samples.extend(subsample_ids)

data_dir = os.path.join(output_path, "rates_data")
os.makedirs(data_dir, exist_ok=True)
# %%
