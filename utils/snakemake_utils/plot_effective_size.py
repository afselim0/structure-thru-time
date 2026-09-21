#!/usr/bin/env python3
"""Plot a per-population effective size proxy, Ne(t) = 1 / (2 * lambda_pop(t)).

lambda_pop(t) is the within-population pairwise coalescent rate, aggregated over
the pairs of that population at each time point. The aggregation happens *before*
the reciprocal (`1 / (2 * mean(lambda))`), not after, because:

  * 1/x is convex, so mean(1/lambda) >= 1/mean(lambda) (Jensen) and the two
    disagree badly exactly where lambda is small, i.e. at recent times;
  * ComputeCoalescentRates floors lambda at ~1e-16 where no coalescence is seen,
    and 1/(2*1e-16) = 5e15. Any "reciprocal first" aggregate inherits that floor
    and diverges (observed: mean(1/2*lambda) reaching 1e15 for chimpanzees),
    whereas an average taken over rates is bounded by the mean rate.

The mean is used rather than the median for the same reason: the median can land
on a floored pair and produce the same divergence, and 1/(2*mean(lambda)) is the
estimator whose interpretation matches PSMC's -- the inverse of the average rate
at which lineages in that population coalesce.
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_style import setup_plotting_style, set_time_xlim

LINE_WIDTH = 2
# Rates at or below this are the estimator's zero-floor, not a measurement, and
# would map to Ne ~ 1e15. Time points where a population is entirely floored are
# left as gaps rather than drawn.
RATE_FLOOR = 1e-15


def compute_effective_sizes(rates_df, pop_labels, per_haplotype=False):
    """Return {population: Ne(t)} plus the time axis, from within-population pairs."""
    id_label = "sample_id" if per_haplotype else "individual_id"
    times = rates_df.columns.astype(float).values
    pairs = list(rates_df.index.values)

    effective_sizes = {}
    for pop in pop_labels["population"].unique():
        ids = set(np.unique(pop_labels[pop_labels["population"] == pop][id_label].values))
        # Within-population pairs only: these are the ones whose coalescent rate
        # reflects that population's own size.
        sel = [p for p in pairs if p[0] in ids and p[1] in ids]
        if len(sel) == 0:
            print(f"  {pop}: no within-population pairs, skipping")
            continue

        rates = rates_df.loc[sel, :].values
        mean_rate = np.nanmean(rates, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ne = 1.0 / (2.0 * mean_rate)
        ne[mean_rate <= RATE_FLOOR] = np.nan
        effective_sizes[pop] = ne
        print(f"  {pop}: {len(sel)} within-population pairs")

    return effective_sizes, times


def plot_effective_sizes(effective_sizes, times, pop_colors, min_time=1,
                         max_Ne_plot=None, time_scale="log", Ne_scale="log",
                         generation_time=None):
    setup_plotting_style()
    color_of = dict(zip(pop_colors["population"], pop_colors["color"]))

    fig, ax = plt.subplots(figsize=(7, 5))
    for pop, ne in effective_sizes.items():
        ax.plot(times, ne, color=color_of.get(pop, "grey"), lw=LINE_WIDTH, label=pop)

    ax.set_xscale(time_scale)
    ax.set_yscale(Ne_scale)
    set_time_xlim(ax, min_time, times, time_scale)
    if max_Ne_plot is not None:
        ax.set_ylim(top=float(max_Ne_plot))
    # A linear axis is conventionally anchored at 0 (as in published PSMC figures);
    # a log axis cannot include 0, so its bottom is left to matplotlib.
    if Ne_scale == "linear":
        ax.set_ylim(bottom=0)
    ax.set_xlabel("Time (generations ago)")
    ax.set_ylabel(r"Effective population size  $1/(2\lambda)$")
    ax.legend(title="Population", bbox_to_anchor=(1.02, 1), loc="upper left")

    # A years axis on top, for comparison with published PSMC curves (which are
    # conventionally plotted in years).
    if generation_time:
        secax = ax.secondary_xaxis("top", functions=(lambda g: g * generation_time,
                                                     lambda y: y / generation_time))
        secax.set_xlabel(f"Years (g = {generation_time:g})")

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot per-population effective size 1/(2*lambda)")
    parser.add_argument("--input", required=True, help="Path to the rates parquet file")
    parser.add_argument("--colors", required=True, help="Path to the population colors csv")
    parser.add_argument("--labels", required=True, help="Path to the population labels csv")
    parser.add_argument("--output_path", required=True, help="Output directory for the plot")
    parser.add_argument("--output_name", required=True, help="Output plot filename")
    parser.add_argument("--min_time", type=float, default=1, help="Left limit of the time axis")
    parser.add_argument("--max_Ne_plot", type=float, default=None, help="Upper limit of the Ne axis")
    parser.add_argument("--time_scale", default="log", help="Time axis scale, linear or log")
    parser.add_argument("--Ne_scale", default="log", help="Effective size (y) axis scale, linear or log")
    parser.add_argument("--generation_time", type=float, default=None,
                        help="Years per generation; adds a secondary years axis when set")
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true",
                        help="If set, pairs are indexed by sample_id rather than individual_id")

    args = parser.parse_args()
    rates_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)

    id_label = "sample_id" if args.per_haplotype else "individual_id"
    if id_label == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2

    # Restrict the labels to the samples the rates were actually computed over,
    # matching plot_rates.py.
    active_samples = np.unique([s for pair in rates_df.index.values for s in pair])
    pop_labels = labels_df[labels_df[id_label].isin(active_samples)]

    print(f"Computing effective sizes from: {args.input}")
    effective_sizes, times = compute_effective_sizes(rates_df, pop_labels,
                                                     per_haplotype=args.per_haplotype)
    if not effective_sizes:
        raise ValueError("No population had within-population pairs; nothing to plot.")

    pop_colors = colors_df[colors_df["population"].isin(effective_sizes.keys())]
    fig = plot_effective_sizes(effective_sizes, times, pop_colors,
                               min_time=args.min_time, max_Ne_plot=args.max_Ne_plot,
                               time_scale=args.time_scale, Ne_scale=args.Ne_scale,
                               generation_time=args.generation_time)

    os.makedirs(args.output_path, exist_ok=True)
    out = os.path.join(args.output_path, args.output_name)
    fig.savefig(out, dpi=500, bbox_inches="tight")
    print(f"saved effective size plot to {out}")


if __name__ == "__main__":
    main()
