"""
Figure data: CoaSTT embeddings on simulated versus SINGER-inferred trees, for two
demographies.

A 2x2 comparison. The rows are the two demographies (a clean two-population split,
and a single population with a sawtooth size history); the columns are the two
tree sources, the simulated tree sequences and the ones SINGER (+POLEGON) inferred
from sequence data simulated on them.

Within a row the two embeddings are only defined up to a rotation/reflection, so the
SINGER embedding is Procrustes-aligned to the simulated-tree one here; the plotting
script draws the frames it is given.

Tables written to results/figure_data/split_and_sawtooth_singer/:

  embeddings.parquet  long form, individual_id, time, X1, X2, demography, source
                      ("simulated trees" / "SINGER trees"), already aligned
  pop_labels.csv      demography, sample_id, individual_id, population
  pop_colors.csv      demography, population, color (taken from the sim run, so a
                      row's two panels share one palette)
  params.csv          demography, parameter, value -- the plotting constants
                      (min_time, max_time, time_scale, xy_lims, d/N_min/N_max/p)
"""
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (CONFIGS_DIR, OUTPUTS_DIR, add_src_to_path, find_output_file,
                     out_dir, report, require, write_table)

add_src_to_path()
from OptimizationFunctions import procrustes_align_trajectories_df

FIGURE = "split_and_sawtooth_singer"

# one row of the figure per entry: the run directory holding the sim/ and singer/
# subdirectories, and the config the embedding and plotting parameters come from
DEMOGRAPHIES = {
    "split": os.path.join(OUTPUTS_DIR, "simulations", "split"),
    "sawtooth": os.path.join(OUTPUTS_DIR, "simulations", "sawtooth"),
}
# label used in the tables and drawn in the column titles
SOURCES = {"sim": "simulated trees", "singer": "SINGER trees"}
REFERENCE_SOURCE = "sim"

D = 2  # every run in this figure is a 2D embedding


def config_for(demography):
    path = os.path.join(CONFIGS_DIR, "sim_configs", f"{demography}.yaml")
    with open(require(path, f"{demography} config")) as handle:
        return yaml.safe_load(handle)


def read_embeddings(run_dir, source, config):
    """The 2D embedding of one (run, tree source), at the config's parameters."""
    data_dir = os.path.join(run_dir, source, "data")
    return pd.read_parquet(find_output_file(
        data_dir, "embeddings", d=D, Nmin=config["N_min"], Nmax=config["N_max"],
        p=config["p"]))


def read_labels(run_dir, source):
    """pop_labels/pop_colors of one run, with the written index column dropped."""
    data_dir = os.path.join(run_dir, source, "data")
    tables = []
    for name in ("pop_labels.csv", "pop_colors.csv"):
        table = pd.read_csv(require(os.path.join(data_dir, name)))
        tables.append(table.loc[:, ~table.columns.str.startswith("Unnamed")])
    return tables


def main():
    destination = out_dir(FIGURE)

    embedding_frames, label_frames, color_frames, param_rows = [], [], [], []

    for demography, run_dir in DEMOGRAPHIES.items():
        config = config_for(demography)

        raw = {source: read_embeddings(run_dir, source, config) for source in SOURCES}

        # the SINGER frame is arbitrary up to an orthogonal transform, so put it in
        # the simulated-tree frame before it is drawn beside it
        reference = raw[REFERENCE_SOURCE]
        for source, label in SOURCES.items():
            df = (reference if source == REFERENCE_SOURCE
                  else procrustes_align_trajectories_df(
                      reference, raw[source], allow_reflections=True))
            embedding_frames.append(
                df.assign(demography=demography, source=label))

        # both panels of a row are coloured by the same palette, read from the run
        # whose frame the row is aligned to
        pop_labels, pop_colors = read_labels(run_dir, REFERENCE_SOURCE)
        label_frames.append(pop_labels.assign(demography=demography))
        color_frames.append(pop_colors.assign(demography=demography))

        for key in ("min_time", "max_time", "time_scale", "N_min", "N_max", "p",
                    "embeddings_plot_lims"):
            param_rows.append({"demography": demography, "parameter": key,
                               "value": config[key]})
        param_rows.append({"demography": demography, "parameter": "d", "value": D})

    embeddings = pd.concat(embedding_frames, ignore_index=True)
    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
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
