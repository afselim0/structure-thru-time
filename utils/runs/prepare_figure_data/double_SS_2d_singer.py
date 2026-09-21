"""
Figure data: the two-group 2D stepping-stone example on simulated versus
SINGER-inferred trees.

The demography is two 5x5 lattices of demes, `g<group>_<row>_<col>`, with one
individual sampled per deme. The figure contrasts the CoaSTT embeddings and the
mean pairwise coalescent rates obtained from the simulated tree sequences with
those obtained from the trees SINGER (+POLEGON) inferred from sequence data
simulated on them.

The SINGER embedding is Procrustes-aligned to the simulated-tree one here, since
the two frames are only defined up to a rotation/reflection.

There is one individual per deme, so no pair of individuals shares a deme and
"within-deme" rates do not exist. The rate panels instead average over two families
of pair classes, both derived here from the deme names:

  neighbour  per group, the pairs whose demes are lattice neighbours in that
             group's grid (|drow| + |dcol| == 1)
  all        every pair within grid g0, every pair within grid g1, and every pair
             spanning the two grids

No analytic rates are computed for this figure: the stepping-stone model's pair
coalescence rates are not what the comparison is about, and the sim run does not
write a rates_true table for it.

Tables written to results/figure_data/double_SS_2d_singer/:

  embeddings.parquet  long form, individual_id, time, X1, X2, source
                      ("simulated trees" / "SINGER trees"), already aligned
  mean_rates.parquet  long form, source, panel ("neighbour"/"all"), pair_class
                      ("g0"/"g1"/"between"), time, rate
  pop_labels.csv      sample_id, individual_id, population
  pop_colors.csv      population, color
  params.csv          parameter, value -- the plotting constants
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

FIGURE = "double_SS_2d_singer"

RUN_DIR = os.path.join(OUTPUTS_DIR, "simulations", "double_2d_SS")
CONFIG_FILE = os.path.join(CONFIGS_DIR, "sim_configs", "double_2d_SS.yaml")

SOURCES = {"simulated trees": "sim", "SINGER trees": "singer"}
REFERENCE_SOURCE = "simulated trees"

D = 2
GROUPS = ["g0", "g1"]
# rates hit exactly zero at the most recent timepoints, which a log axis cannot
# show; the figure floors them here instead
RATE_FLOOR = 1e-7


def data_dir(source):
    return os.path.join(RUN_DIR, SOURCES[source], "data")


def read_config():
    with open(require(CONFIG_FILE, "double_2d_SS config")) as handle:
        return yaml.safe_load(handle)


def read_labels():
    directory = data_dir(REFERENCE_SOURCE)
    tables = []
    for name in ("pop_labels.csv", "pop_colors.csv"):
        table = pd.read_csv(require(os.path.join(directory, name)))
        tables.append(table.loc[:, ~table.columns.str.startswith("Unnamed")])
    return tables


def deme_coordinates(individual_pop):
    """
    Split each individual's deme name `g<group>_<row>_<col>` into its parts.

    Returns a frame indexed by individual_id with group/row/col columns, which is
    what the lattice adjacency below is derived from.
    """
    parts = individual_pop.astype(str).str.split("_", expand=True)
    if parts.shape[1] != 3:
        raise ValueError(
            "deme names are expected to be g<group>_<row>_<col>, got e.g. "
            f"{individual_pop.iloc[0]!r}")
    return pd.DataFrame({"group": parts[0],
                         "row": parts[1].astype(int),
                         "col": parts[2].astype(int)},
                        index=individual_pop.index)


def pair_masks(rates_index, coordinates):
    """
    Boolean mask per pair class, on the rows of a rates table.

    Two families of classes, keyed (panel, pair_class):
      ("neighbour", g)   both individuals in grid g, their demes one lattice step
                         apart
      ("all", g)         both individuals in grid g, any two demes
      ("all", "between") the two individuals in different grids
    """
    left = coordinates.reindex(rates_index.get_level_values(0))
    right = coordinates.reindex(rates_index.get_level_values(1))
    if left.isna().any().any() or right.isna().any().any():
        raise ValueError("some rate pairs reference individuals missing from pop_labels")

    same_group = (left["group"].to_numpy() == right["group"].to_numpy())
    lattice_distance = (np.abs(left["row"].to_numpy() - right["row"].to_numpy())
                        + np.abs(left["col"].to_numpy() - right["col"].to_numpy()))

    masks = {}
    for group in GROUPS:
        in_group = same_group & (left["group"].to_numpy() == group)
        masks[("neighbour", group)] = in_group & (lattice_distance == 1)
        masks[("all", group)] = in_group
    masks[("all", "between")] = ~same_group

    for key, mask in masks.items():
        if not mask.any():
            raise ValueError(f"no individual pairs fall in the class {key}")
    return masks


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
    coordinates = deme_coordinates(individual_pop)

    # ---- the class means, per source ----
    mean_frames = []
    for source in SOURCES:
        rates = pd.read_parquet(find_output_file(data_dir(source), "rates"))
        times = rates.columns.to_numpy(dtype=float)
        values = rates.to_numpy(dtype=float)
        masks = pair_masks(rates.index, coordinates)
        for (panel, pair_class), mask in masks.items():
            mean_frames.append(pd.DataFrame({
                "source": source, "panel": panel, "pair_class": pair_class,
                "time": times, "rate": values[mask].mean(axis=0)}))
    mean_rates = pd.concat(mean_frames, ignore_index=True)

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
        write_table(mean_rates, os.path.join(destination, "mean_rates.parquet")),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
