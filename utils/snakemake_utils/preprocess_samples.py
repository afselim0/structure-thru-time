#%%
#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

def preprocess_samples(pop_labels_path: str, pop_colors_path: str, pops: list[str] | None,
                        n_samples_per_pop: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    population_labels = pd.read_csv(pop_labels_path)
    population_colors = pd.read_csv(pop_colors_path)
    # Drop stray index columns left over from previous double-indexed to_csv saves
    population_labels = population_labels.loc[:, ~population_labels.columns.str.match(r"^Unnamed")]
    population_colors = population_colors.loc[:, ~population_colors.columns.str.match(r"^Unnamed")]

    all_pops = population_labels["population"].unique()
    if pops is not None:
        missing_pops = np.setdiff1d(pops, all_pops)
        if len(missing_pops) > 0:
            raise ValueError(f"Some chosen populations are not in the population labels file: {missing_pops.tolist()}")
        population_labels = population_labels[population_labels["population"].isin(pops)]
        population_colors = population_colors[population_colors["population"].isin(pops)]

    if n_samples_per_pop is not None:
        rng = np.random.default_rng(seed)
        chosen_samples = []
        for pop in population_labels["population"].unique():
            sample_ids = population_labels[population_labels["population"] == pop]["sample_id"].values
            if n_samples_per_pop < len(sample_ids):
                subsample_ids = rng.choice(sample_ids, size=n_samples_per_pop, replace=False)
            else:
                subsample_ids = sample_ids
            chosen_samples.extend(subsample_ids)
        population_labels = population_labels[population_labels["sample_id"].isin(chosen_samples)]

    population_labels = population_labels.sort_values("sample_id").reset_index(drop=True)
    population_colors = population_colors.reset_index(drop=True)
    return population_labels, population_colors

def main():
    parser = argparse.ArgumentParser(description="Preprocess population labels and colors for an empirical (non-simulation) run: "
                                                   "subsets the provided pop_labels/pop_colors files to the populations and sample "
                                                   "counts requested for this run, so downstream scripts consume pre-sliced inputs.")
    parser.add_argument("--pop_labels", required=True, help="Path to the original population labels csv file", type=str)
    parser.add_argument("--pop_colors", required=True, help="Path to the original population colors csv file", type=str)
    parser.add_argument("--output_path", required=True, help="Output directory for the sliced pop_labels/pop_colors files", type=str)
    parser.add_argument("--pops", nargs="+", default=None, help="Population labels to keep. If omitted, all populations are kept.")
    parser.add_argument("--n_samples_per_pop", type=int, default=-1, help="Number of samples to randomly draw per population. If -1, all samples are kept.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used when subsampling with --n_samples_per_pop")

    args = parser.parse_args()
    pops = None if args.pops is None or args.pops == ["all"] else args.pops
    n_samples_per_pop = None if args.n_samples_per_pop == -1 else args.n_samples_per_pop

    print(f"preprocessing samples from {args.pop_labels} (pops={pops or 'all'}, n_samples_per_pop={n_samples_per_pop or 'all'})")
    population_labels, population_colors = preprocess_samples(
        pop_labels_path=args.pop_labels,
        pop_colors_path=args.pop_colors,
        pops=pops,
        n_samples_per_pop=n_samples_per_pop,
        seed=args.seed,
    )
    os.makedirs(args.output_path, exist_ok=True)
    population_labels.to_csv(os.path.join(args.output_path, "pop_labels.csv"), index=False)
    population_colors.to_csv(os.path.join(args.output_path, "pop_colors.csv"), index=False)
    print(f"kept {len(population_labels)} samples across {population_labels['population'].nunique()} populations")

if __name__ == "__main__":
    main()
