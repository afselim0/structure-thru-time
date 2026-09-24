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
              "max_window_size", "pdf_max_time", "pdf_bin_size", "rates_min_time",
              "rates_max_time", "time_scale", "embeddings_min_time", "max_rate_plot")
# the snakefile's defaults for keys a config may leave out
PARAM_DEFAULTS = {"embeddings_min_time": "rates_min_time"}
# sources whose between/within ratio crossing 0.5 is reported as a split-time estimate
HALF_RATIO_DEMOGRAPHY = "split"
HALF_RATIO_SOURCES = ("tsinfer", "relate", "singer")
# fraction of the true log peak-trough contrast each inferred-ARG source retains
CONTRAST_DEMOGRAPHY = "sawtooth"
CONTRAST_SOURCES = HALF_RATIO_SOURCES
# seeds are fixed per contig i by the pipeline, not drawn here
SEED_RULE = "mutations: i+42 (infer_trees.py); simulation: 42+i (simulate_trees.py); SINGER/Relate: i+1"


def read_rates(run_root, demography, source, config):
    """Wide rates for one (demography, source), with float time columns."""
    data_dir = os.path.join(run_root, demography, SOURCES[source], "data")
    # rates_true carries a tag now, so it is looked up like any other output.
    path = (find_output_file(data_dir, "rates_true") if source == "true"
            else find_output_file(data_dir, "rates",
                                  minWS=config["min_window_size"],
                                  maxWS=config["max_window_size"]))
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


def first_crossing(times, ratio, level=0.5):
    """First time (going back) where ratio >= level, linearly interpolated."""
    above = np.flatnonzero(ratio >= level)
    if above.size == 0:
        return np.nan
    i = above[0]
    if i == 0:
        return times[0]
    t0, t1, r0, r1 = times[i - 1], times[i], ratio[i - 1], ratio[i]
    return t0 + (level - r0) * (t1 - t0) / (r1 - r0)


def sawtooth_extrema(demes_path):
    """Interior rate extrema of a one-deme sawtooth, oldest and youngest breakpoint
    excluded, as (time, kind, search_lo, search_hi): each curve's own extremum is
    looked for between the log-midpoints to the neighbouring breakpoints."""
    with open(demes_path) as f:
        epochs = yaml.safe_load(f)["demes"][0]["epochs"]
    breaks = sorted((e["end_time"], e.get("end_size", e["start_size"]))
                    for e in epochs if e["end_time"] > 0)
    times = np.array([t for t, _ in breaks])
    sizes = np.array([size for _, size in breaks])
    mids = np.sqrt(times[:-1] * times[1:])
    return [(times[i], "trough" if sizes[i] > np.median(sizes) else "peak", mids[i - 1], mids[i])
            for i in range(1, len(times) - 1)]


def curve_extremum(rate, kind, lo, hi):
    """Time and value of a curve's own max (peak) or min (trough) within [lo, hi],
    and whether it is not a turn of the full curve (still climbing past the window)."""
    window = rate[(rate.index >= lo) & (rate.index <= hi)]
    t = window.idxmax() if kind == "peak" else window.idxmin()
    i = rate.index.get_loc(t)
    neighbours = rate.iloc[max(i - 1, 0):i + 2]
    turn = neighbours.max() == rate[t] if kind == "peak" else neighbours.min() == rate[t]
    return t, rate[t], not turn


def contrast_retained(means, extrema):
    """rho = log(lhat(a) / lhat(b)) / log(l(a) / l(b)) for consecutive extrema a, b
    (youngest first), each curve evaluated at its own extremum."""
    within = means[means["pair_type"] == "within"]
    curves = {s: g.set_index("time")["rate"].sort_index() for s, g in within.groupby("source")}
    true = {k: curve_extremum(curves["true"], kind, lo, hi)[1]
            for k, (_, kind, lo, hi) in enumerate(extrema)}
    rows = []
    for source in CONTRAST_SOURCES:
        found = [curve_extremum(curves[source], kind, lo, hi) for _, kind, lo, hi in extrema]
        for k in range(len(extrema) - 1):
            (t_a, v_a, edge_a), (t_b, v_b, edge_b) = found[k], found[k + 1]
            if edge_a or edge_b:
                print(f"  note: {source} contrast {k + 1} extremum is not a turn of the curve")
            rows.append({"source": source, "contrast": k + 1,
                         "time_a": extrema[k][0], "kind_a": extrema[k][1],
                         "time_b": extrema[k + 1][0], "kind_b": extrema[k + 1][1],
                         "curve_time_a": t_a, "curve_time_b": t_b,
                         "not_a_turn": edge_a or edge_b,
                         "rho": np.log(v_a / v_b) / np.log(true[k] / true[k + 1])})
    return pd.DataFrame(rows)


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
    ratio_rows = []
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
        reference = read_rates(args.root, demography, "sim_trees", config)

        pairs = pair_table(reference.index, pop_labels)
        pairs.insert(0, "demography", demography)
        pair_frames.append(pairs)
        pair_type = pairs["pair_type"].to_numpy()

        for source in SOURCES:
            rates = read_rates(args.root, demography, source, config)
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
            group_means = {}
            for group in np.unique(pair_type):
                mean = rates.to_numpy()[pair_type == group].mean(axis=0)
                group_means[group] = mean
                mean_rows.append(pd.DataFrame({
                    "demography": demography, "source": source,
                    "pair_type": group, "time": times, "rate": mean}))
            if {"within", "between"} <= group_means.keys():
                ratio_rows.append(pd.DataFrame({
                    "demography": demography, "source": source, "time": times,
                    "ratio": group_means["between"] / group_means["within"]}))

        for key in PARAM_KEYS:
            value = config.get(key, config.get(PARAM_DEFAULTS.get(key)))
            param_rows.append({"demography": demography, "parameter": key,
                               "value": value})
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
    ratios = pd.concat(ratio_rows, ignore_index=True)
    written.append(write_table(
        ratios, os.path.join(destination, "between_within_ratio.parquet")))
    half_rows = []
    for source in HALF_RATIO_SOURCES:
        r = ratios[(ratios["demography"] == HALF_RATIO_DEMOGRAPHY)
                   & (ratios["source"] == source)].sort_values("time")
        half_rows.append({"demography": HALF_RATIO_DEMOGRAPHY, "source": source,
                          "time": first_crossing(r["time"].to_numpy(),
                                                 r["ratio"].to_numpy())})
    half_times = pd.DataFrame(half_rows)
    contrast_config = os.path.join(args.root, CONTRAST_DEMOGRAPHY, "configs", "sim.yaml")
    with open(contrast_config) as f:
        demes_path = os.path.join(REPO_ROOT, "utils", yaml.safe_load(f)["demes_file_path"])
    means = pd.concat(mean_rows, ignore_index=True)
    contrast = contrast_retained(means[means["demography"] == CONTRAST_DEMOGRAPHY],
                                 sawtooth_extrema(require(demes_path, "sawtooth demes file")))
    written.append(write_table(contrast, os.path.join(destination, "contrast_retained.csv"),
                               index=False))
    written.append(write_table(half_times, os.path.join(destination, "ratio_half_times.csv"),
                               index=False))
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
