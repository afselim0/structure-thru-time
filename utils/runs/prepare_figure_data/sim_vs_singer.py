"""
Figure data: CoaSTT embeddings on simulated versus SINGER-inferred trees, for four
demographies (split, sawtooth, complex_4_pops, double_2D_SS).

Within a demography the SINGER embedding is Procrustes-aligned onto the
simulated-tree embedding (reference frame: its first timepoint, matched by
individual_id). The double_2D_SS demes are recoloured here with one sequential
colormap per lattice group, as in the double_SS_2d figure.

Tables written to results/figure_data/sim_vs_singer/:

  embeddings.parquet  individual_id, time, X1, X2, demography, source
                      ("simulated trees" / "SINGER trees"), already aligned
  pop_labels.csv      demography, sample_id, individual_id, population
  pop_colors.csv      demography, population, color
  params.csv          demography, parameter, value (min_time, max_time, time_scale,
                      xy_lims, N_min, N_max, p, d)
"""
import os
import sys

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (CONFIGS_DIR, OUTPUTS_DIR, add_src_to_path, find_output_file,
                     out_dir, report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "sim_vs_singer"

SIM_DIR = os.path.join(OUTPUTS_DIR, "simulations")
# demography -> (run directory holding sim/ and singer/, config name)
DEMOGRAPHIES = {
    "split": (os.path.join(SIM_DIR, "split"), "split"),
    "sawtooth": (os.path.join(SIM_DIR, "sawtooth"), "sawtooth"),
    "complex_4_pops": (os.path.join(SIM_DIR, "complex_4_pops", "complex_4_pops_m_0"),
                       "complex_4_pops"),
    "double_2D_SS": (os.path.join(SIM_DIR, "double_2d_SS"), "double_2d_SS"),
}
SOURCES = {"sim": "simulated trees", "singer": "SINGER trees"}
REFERENCE_SOURCE = "sim"
D = 2
XY_PAD = 1.05
TIME_SCALE = {"split": "log"}

GROUP_CMAPS = {"g0": "spring", "g1": "winter"}
CMAP_RANGE = (0.15, 0.95)


def read_config(name):
    path = os.path.join(CONFIGS_DIR, "sim_configs", f"{name}.yaml")
    with open(require(path, f"{name} config")) as handle:
        return yaml.safe_load(handle)


def read_embeddings(run_dir, source, config):
    return pd.read_parquet(find_output_file(
        os.path.join(run_dir, source, "data"), "embeddings", d=D,
        Nmin=config["N_min"], Nmax=config["N_max"], p=config.get("p", 1)))


def read_labels(run_dir, source):
    data_dir = os.path.join(run_dir, source, "data")
    tables = []
    for name in ("pop_labels.csv", "pop_colors.csv"):
        table = pd.read_csv(require(os.path.join(data_dir, name)))
        tables.append(table.loc[:, ~table.columns.str.startswith("Unnamed")])
    return tables


def xy_limit(frames):
    radius = max(float(np.hypot(df["X1"], df["X2"]).max()) for df in frames)
    padded = radius * XY_PAD
    magnitude = 10 ** np.floor(np.log10(padded) - 1)
    return float(np.ceil(padded / magnitude) * magnitude)


def group_colors(pop_colors):
    pop_colors = pop_colors.copy()
    groups = pop_colors["population"].str.split("_").str[0]
    for group, cmap_name in GROUP_CMAPS.items():
        members = groups == group
        cmap = matplotlib.colormaps[cmap_name]
        shades = np.linspace(*CMAP_RANGE, members.sum())
        pop_colors.loc[members, "color"] = [mcolors.to_hex(cmap(s)[:3]) for s in shades]
    return pop_colors


def main():
    destination = out_dir(FIGURE)
    embedding_frames, label_frames, color_frames, param_rows = [], [], [], []

    for demography, (run_dir, config_name) in DEMOGRAPHIES.items():
        config = read_config(config_name)
        raw = {s: read_embeddings(run_dir, s, config) for s in SOURCES}
        reference = raw[REFERENCE_SOURCE]
        aligned = {s: (reference if s == REFERENCE_SOURCE
                       else procrustes_align_trajectories_df(
                           reference, raw[s], allow_reflections=True))
                   for s in SOURCES}
        for s, df in aligned.items():
            embedding_frames.append(df.assign(demography=demography, source=SOURCES[s]))

        pop_labels, pop_colors = read_labels(run_dir, REFERENCE_SOURCE)
        if demography == "double_2D_SS":
            pop_colors = group_colors(pop_colors)
        label_frames.append(pop_labels.assign(demography=demography))
        color_frames.append(pop_colors.assign(demography=demography))

        values = {
            "min_time": config.get("embeddings_min_time", config["rates_min_time"]),
            "max_time": config.get("embeddings_max_time", config["rates_max_time"]),
            "time_scale": TIME_SCALE.get(demography, config["time_scale"]),
            "xy_lims": xy_limit(aligned.values()),
            "N_min": config["N_min"], "N_max": config["N_max"],
            "p": config.get("p", 1), "d": D,
        }
        param_rows += [{"demography": demography, "parameter": k, "value": v}
                       for k, v in values.items()]

    written = [
        write_table(pd.concat(embedding_frames, ignore_index=True),
                    os.path.join(destination, "embeddings.parquet")),
        write_table(pd.concat(label_frames, ignore_index=True),
                    os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pd.concat(color_frames, ignore_index=True),
                    os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(pd.DataFrame(param_rows),
                    os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
