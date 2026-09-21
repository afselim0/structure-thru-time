"""
Figure data: the complex four-population example.

Collects what the figure's four panels need from the complex_4_pops run:
  A) the demography is drawn straight from the demes file by the plotting script,
     so only the population colours and the time-axis settings are exported here
  B) the embeddings through time, with the population labels/colours they are
     coloured by
  C) the estimated and reconstructed rates, whose agreement the figure histograms
  D) the p scan, i.e. the loss and the correlations against the true rates as the
     link's exponent p is varied
"""
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (CONFIGS_DIR, OUTPUTS_DIR, find_output_file, out_dir, report,
                     require, write_table)

FIGURE = "complex_4_pops"

RUN_DIR = os.path.join(OUTPUTS_DIR, "simulations", "complex_4_pops", "complex_4_pops_m_0", "sim", "data")
CONFIG_FILE = os.path.join(CONFIGS_DIR, "sim_configs", "complex_4_pops.yaml")

# the embedding this figure shows, and the N_max its rate panels are shifted by
D, N_MIN, N_MAX, P = 2, 1000, 1000000, 1


def main():
    destination = out_dir(FIGURE)

    embeddings = pd.read_parquet(
        find_output_file(RUN_DIR, "embeddings", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
    estimated_rates = pd.read_parquet(find_output_file(RUN_DIR, "rates"))
    reconstructed_rates = pd.read_parquet(
        find_output_file(RUN_DIR, "reconstructed_rates", d=D, Nmin=N_MIN, Nmax=N_MAX, p=P))
    scan_p = pd.read_csv(
        find_output_file(RUN_DIR, "scan_p", d=D, Nmin=N_MIN, Nmax=N_MAX))

    pop_labels = pd.read_csv(require(os.path.join(RUN_DIR, "pop_labels.csv")))
    pop_colors = pd.read_csv(require(os.path.join(RUN_DIR, "pop_colors.csv")))
    # the run writes these with an unnamed index column; drop it so the tables the
    # figure reads carry only real columns
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]
    pop_colors = pop_colors.loc[:, ~pop_colors.columns.str.startswith("Unnamed")]

    with open(require(CONFIG_FILE)) as handle:
        config = yaml.safe_load(handle)
    # the plotting parameters the figure takes from the run's config, so the plot
    # script does not have to open a pipeline config of its own
    plot_params = pd.DataFrame([
        # The emitted names stay min_time/max_time: they are this figure's plot-axis
        # contract with its plotting script, not the pipeline's pdf grid.
        {"parameter": "min_time", "value": config["embeddings_min_time"]},
        {"parameter": "max_time", "value": config.get("embeddings_max_time", config["pdf_max_time"])},
        {"parameter": "time_scale", "value": config["time_scale"]},
        {"parameter": "N_max", "value": N_MAX},
    ])

    written = [
        write_table(embeddings, os.path.join(destination, "embeddings.parquet")),
        write_table(estimated_rates, os.path.join(destination, "estimated_rates.parquet")),
        write_table(reconstructed_rates, os.path.join(destination, "reconstructed_rates.parquet")),
        write_table(scan_p, os.path.join(destination, "scan_p.csv"), index=False),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(plot_params, os.path.join(destination, "plot_params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
