"""
Figure data: the complex four-population example on simulated versus
SINGER-inferred trees.

The same demography is run twice, once on the simulated tree sequences and once on
the trees SINGER (+POLEGON) inferred from sequence data simulated on them. The
figure contrasts the two, both as embeddings and as the pairwise coalescent rates
the embeddings are fitted to, with the analytic rates of the demes model as the
reference the two estimates are judged against.

The embeddings are defined only up to a rotation/reflection, so the SINGER frame is
Procrustes-aligned to the simulated-tree frame here; the plotting script draws the
frames it is given.

The rate panels show group means rather than individual pairs: for each population
the mean over its within-population individual pairs, and for each pair of
populations the mean over the pairs spanning them. The three sources are kept on
their own time grids (the analytic grid is an np.linspace while the estimates use a
max_time/n_timepoints step, so they differ in the last digits) and drawn as separate
lines, which needs no alignment.

Tables written to results/figure_data/complex_4_pops_singer/:

  embeddings.parquet  long form, individual_id, time, X1, X2, source
                      ("simulated trees" / "SINGER trees"), already aligned
  mean_rates.parquet  long form, source ("analytic truth" / "simulated trees" /
                      "SINGER trees"), group ("within"/"between"), population_1,
                      population_2, time, rate
  pop_labels.csv      sample_id, individual_id, population
  pop_colors.csv      population, color
  params.csv          parameter, value -- the plotting constants (time window and
                      scale, embedding parameters, rate axis limits)
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (CONFIGS_DIR, OUTPUTS_DIR, add_src_to_path, find_output_file,
                     out_dir, report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "complex_4_pops_singer"

RUN_DIR = os.path.join(OUTPUTS_DIR, "simulations", "complex_4_pops",
                       "complex_4_pops_m_0")
CONFIG_FILE = os.path.join(CONFIGS_DIR, "sim_configs", "complex_4_pops.yaml")

# tree source -> the run subdirectory it lives in, labelled as the figure names it
SOURCES = {"simulated trees": "sim", "SINGER trees": "singer"}
REFERENCE_SOURCE = "simulated trees"
# the analytic rates the demes model implies; only the sim run writes them
TRUTH_SOURCE = "analytic truth"

D = 2  # every run in this figure is a 2D embedding
# the between-population analytic rates are exactly zero before the populations
# merge, which a log axis cannot show; the figure floors them here instead
RATE_FLOOR = 1e-6


def data_dir(source):
    return os.path.join(RUN_DIR, SOURCES[source], "data")


def read_config():
    with open(require(CONFIG_FILE, "complex_4_pops config")) as handle:
        return yaml.safe_load(handle)


def read_labels():
    """pop_labels/pop_colors of the reference run, with the written index dropped."""
    directory = data_dir(REFERENCE_SOURCE)
    tables = []
    for name in ("pop_labels.csv", "pop_colors.csv"):
        table = pd.read_csv(require(os.path.join(directory, name)))
        tables.append(table.loc[:, ~table.columns.str.startswith("Unnamed")])
    return tables


def group_means(rates, individual_pop, populations):
    """
    Mean rate per population and per population pair, long form.

    Rates are stored one row per individual pair, so a within-population mean
    averages the pairs whose two individuals share a population and a
    between-population mean averages the pairs spanning the two given ones. There
    are no self pairs, so a within-population mean is over distinct individuals.
    """
    pop_1 = individual_pop.reindex(rates.index.get_level_values(0)).to_numpy()
    pop_2 = individual_pop.reindex(rates.index.get_level_values(1)).to_numpy()
    if pd.isna(pop_1).any() or pd.isna(pop_2).any():
        raise ValueError("some rate pairs reference individuals missing from pop_labels")

    times = rates.columns.to_numpy(dtype=float)
    values = rates.to_numpy(dtype=float)

    rows = []
    for i, population_1 in enumerate(populations):
        for population_2 in populations[i:]:
            if population_1 == population_2:
                selected = (pop_1 == population_1) & (pop_2 == population_1)
                group = "within"
            else:
                selected = (((pop_1 == population_1) & (pop_2 == population_2))
                            | ((pop_1 == population_2) & (pop_2 == population_1)))
                group = "between"
            if not selected.any():
                raise ValueError(
                    f"no individual pairs for {population_1}-{population_2}")
            rows.append(pd.DataFrame({
                "group": group,
                "population_1": population_1,
                "population_2": population_2,
                "time": times,
                "rate": values[selected].mean(axis=0)}))
    return pd.concat(rows, ignore_index=True)


def main():
    destination = out_dir(FIGURE)
    config = read_config()

    # ---- embeddings, the SINGER frame put in the simulated-tree frame ----
    raw = {}
    for source in SOURCES:
        raw[source] = pd.read_parquet(find_output_file(
            data_dir(source), "embeddings", d=D, Nmin=config["N_min"],
            Nmax=config["N_max"], p=config["p"]))

    reference = raw[REFERENCE_SOURCE]
    embedding_frames = []
    for source in SOURCES:
        df = (reference if source == REFERENCE_SOURCE
              else procrustes_align_trajectories_df(
                  reference, raw[source], allow_reflections=True))
        embedding_frames.append(df.assign(source=source))
    embeddings = pd.concat(embedding_frames, ignore_index=True)

    pop_labels, pop_colors = read_labels()
    individual_pop = (pop_labels.drop_duplicates("individual_id")
                      .set_index("individual_id")["population"])
    # a stable population order, so the figure's colours and pair labels do not
    # depend on the order the labels file happens to be written in
    populations = sorted(pop_colors["population"].astype(str).unique())

    # ---- group-mean rates: the analytic truth and the two estimates ----
    rate_tables = {TRUTH_SOURCE: pd.read_parquet(require(
        os.path.join(data_dir(REFERENCE_SOURCE), "rates_true.parquet"),
        "analytic rates of the sim run"))}
    for source in SOURCES:
        rate_tables[source] = pd.read_parquet(
            find_output_file(data_dir(source), "rates"))

    mean_frames = []
    for source, rates in rate_tables.items():
        rates = rates.copy()
        rates.columns = rates.columns.astype(float)
        means = group_means(rates, individual_pop, populations)
        mean_frames.append(means.assign(source=source))
    mean_rates = pd.concat(mean_frames, ignore_index=True)
    mean_rates = mean_rates[["source", "group", "population_1", "population_2",
                             "time", "rate"]]

    params = pd.DataFrame([
        # The emitted names stay min_time/max_time: they are this figure's plot-axis
        # contract with its plotting script, not the pipeline's pdf grid.
        {"parameter": "min_time", "value": config["embeddings_min_time"]},
        {"parameter": "max_time", "value": config.get("embeddings_max_time", config["pdf_max_time"])},
        {"parameter": "time_scale", "value": config["time_scale"]},
        {"parameter": "embeddings_plot_lims", "value": config["embeddings_plot_lims"]},
        {"parameter": "max_rate_plot", "value": config["max_rate_plot"]},
        {"parameter": "rate_floor", "value": RATE_FLOOR},
        {"parameter": "N_min", "value": config["N_min"]},
        {"parameter": "N_max", "value": config["N_max"]},
        {"parameter": "p", "value": config["p"]},
        {"parameter": "d", "value": D},
    ])

    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(mean_rates.astype({"rate": np.float64}),
                    os.path.join(destination, "mean_rates.parquet")),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
