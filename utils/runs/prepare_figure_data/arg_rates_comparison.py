"""
Figure data: pairwise coalescent rates across ARG inference methods.

For each of two demographies (split, sawtooth) the comparison holds five rate
sources on the same simulated contigs:

    true       the analytic pair coalescence rates implied by the demes model
    sim_trees  CoaSTT's estimate on the true (simulated) tree sequences
    tsinfer    the estimate on trees inferred with tsinfer + tsdate
    relate     the estimate on trees inferred with Relate
    singer     the estimate on trees inferred with SINGER + POLEGON

The runs are produced by utils/misc_utils/arg_rates_comparison.py (see
utils/runs/local_runs/app_runs/arg_rates_comparison_slurm.sh). The analytic truth
is now computed on the same rates grid as the estimates (compute_rates_true builds
it with rates_time_grid), so the two agree exactly. align_time_columns is still
applied -- it also serves the reconstructed-rates path, where the embedding grid is
genuinely coarser -- and this script insists the whole grid aligns.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (REPO_ROOT, add_src_to_path, find_output_file, out_dir,
                     report, require, write_table)

add_src_to_path()
from evaluate_performance import align_time_columns

FIGURE = "arg_rates_comparison"
DEFAULT_ROOT = os.path.join(REPO_ROOT, "results", "outputs", "misc_analyses",
                            FIGURE)

DEMOGRAPHIES = ("split", "sawtooth")
# source -> the run subdirectory it lives in ("sim" for both true and sim_trees)
SOURCES = {
    "true": "sim",
    "sim_trees": "sim",
    "tsinfer": "tsinfer",
    "relate": "relate",
    "singer": "singer",
}
# the plotting constants each demography's config carries through to the figure
# Read straight out of the config, so these are the CONFIG key names; the figure's
# plotting script reads them back under these same names.
PARAM_KEYS = ("num_contigs", "contig_length", "n_sim_per_pop", "min_window_size",
              "max_window_size", "pdf_max_time", "pdf_bin_size", "time_scale",
              "embeddings_min_time", "max_rate_plot")
# seeds are fixed per contig i by the pipeline, not drawn here
SEED_RULE = "mutations: i+42 (infer_trees.py); simulation: 42+i (simulate_trees.py); SINGER/Relate: i+1"


def read_rates(run_root, demography, source):
    """Wide rates for one (demography, source), with float time columns."""
    data_dir = os.path.join(run_root, demography, SOURCES[source], "data")
    # rates_true carries a tag now, so it is looked up like any other output.
    path = (find_output_file(data_dir, "rates_true") if source == "true"
            else find_output_file(data_dir, "rates"))
    rates = pd.read_parquet(require(path, f"{demography}/{source} rates"))
    rates.columns = rates.columns.astype(float)
    return rates


def pair_table(rates_index, pop_labels):
    """Per-pair populations and within/between type, on the rates' pair index."""
    individual_pop = (pop_labels.drop_duplicates("individual_id")
                      .set_index("individual_id")["population"])
    pop_1 = individual_pop.reindex(rates_index.get_level_values(0)).to_numpy()
    pop_2 = individual_pop.reindex(rates_index.get_level_values(1)).to_numpy()
    if pd.isna(pop_1).any() or pd.isna(pop_2).any():
        raise ValueError("some rate pairs reference individuals missing from pop_labels")
    return pd.DataFrame(
        {"individual_1": rates_index.get_level_values(0),
         "individual_2": rates_index.get_level_values(1),
         "population_1": pop_1,
         "population_2": pop_2,
         "pair_type": np.where(pop_1 == pop_2, "within", "between")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=DEFAULT_ROOT,
                   help="run root holding <demography>/<source>/ (default: %(default)s)")
    p.add_argument("--figure_name", default=FIGURE,
                   help="results/figure_data/<figure_name>/ to write into")
    args = p.parse_args()

    destination = out_dir(args.figure_name)
    written = []
    mean_rows = []
    pair_frames = []
    param_rows = []

    for demography in DEMOGRAPHIES:
        config_path = os.path.join(args.root, demography, "configs", "sim.yaml")
        with open(require(config_path, f"{demography} derived config")) as f:
            config = yaml.safe_load(f)

        pop_labels = pd.read_csv(require(os.path.join(
            args.root, demography, "sim", "data", "pop_labels.csv")))
        pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]

        # the estimate on the true trees defines the time grid every source is put on
        reference = read_rates(args.root, demography, "sim_trees")

        pairs = pair_table(reference.index, pop_labels)
        pairs.insert(0, "demography", demography)
        pair_frames.append(pairs)
        pair_type = pairs["pair_type"].to_numpy()

        for source in SOURCES:
            rates = read_rates(args.root, demography, source)
            if source != "sim_trees":
                rates, aligned_reference = align_time_columns(rates, reference, source)
                if aligned_reference.shape[1] != reference.shape[1]:
                    raise ValueError(
                        f"{demography}/{source}: only {aligned_reference.shape[1]} of "
                        f"{reference.shape[1]} timepoints aligned to the reference grid")
            if not rates.index.equals(reference.index):
                rates = rates.reindex(reference.index)
                if rates.isna().all(axis=1).any():
                    raise ValueError(f"{demography}/{source}: rate table is missing pairs")

            written.append(write_table(
                rates.astype(np.float32),
                os.path.join(destination, f"rates_{demography}_{source}.parquet")))

            times = rates.columns.values.astype(float)
            for group in np.unique(pair_type):
                mean = rates.to_numpy()[pair_type == group].mean(axis=0)
                mean_rows.append(pd.DataFrame({
                    "demography": demography, "source": source,
                    "pair_type": group, "time": times, "rate": mean}))

        for key in PARAM_KEYS:
            param_rows.append({"demography": demography, "parameter": key,
                               "value": config.get(key)})
        param_rows.append({"demography": demography, "parameter": "n_pairs",
                           "value": len(reference)})
        param_rows.append({"demography": demography, "parameter": "n_timepoints_out",
                           "value": reference.shape[1]})
        param_rows.append({"demography": demography, "parameter": "seed_rule",
                           "value": SEED_RULE})

    # long-form group means: one row per (demography, source, pair_type, time)
    written.append(write_table(
        pd.concat(mean_rows, ignore_index=True).reset_index(drop=True),
        os.path.join(destination, "mean_rates.parquet")))
    written.append(write_table(pd.concat(pair_frames, ignore_index=True),
                               os.path.join(destination, "pairs.csv"), index=False))
    written.append(write_table(pd.DataFrame(param_rows),
                               os.path.join(destination, "params.csv"), index=False))
    report(written)

    print("\n  table sizes:")
    for path in written:
        print(f"    {os.path.basename(path):<40} {os.path.getsize(path) / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
