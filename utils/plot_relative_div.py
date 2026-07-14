#!/usr/bin/env python3
"""
Plot between-population coalescence rate relative to the weighted within-population mean.

For each population pair (A, B) at each time t:

    rho_AB(t) = mean_R_AB(t) / mean_R_within(t)

where mean_R_within(t) is the mean over all within-A and within-B pairs pooled,
weighted by the number of pairs each population contributes.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations

from plotting_style import setup_plotting_style

INDIVIDUAL_LINE_WIDTH = 0.2
INDIVIDUAL_LINE_ALPHA = 0.2
MEAN_LINE_WIDTH = 2
MEAN_LINE_ALPHA = 1


def get_pairs(samples1, samples2, within=False):
    """Return sorted (s1, s2) pairs with s1 < s2 between two sample lists.

    If within=True, samples1 == samples2 and we return all within-group pairs.
    """
    pairs = []
    if within:
        for s1, s2 in combinations(samples1, 2):
            pair = (min(s1, s2), max(s1, s2))
            if pair not in pairs:
                pairs.append(pair)
    else:
        for s1 in samples1:
            for s2 in samples2:
                if s1 != s2:
                    pair = (min(s1, s2), max(s1, s2))
                    if pair not in pairs:
                        pairs.append(pair)
    return pairs


def compute_relative_divergence(rates_df, pop_labels, id_col="sample_id"):
    """Compute rho_AB(t) for every off-diagonal population pair (A, B).

    Returns a dict keyed by (popA, popB) with values being a dict:
        "between_pairs":  list of (s1, s2) between-pop pairs
        "mean_between":   np.ndarray (T,)
        "mean_within":    np.ndarray (T,)
        "rho":            np.ndarray (T,)  — rho_AB = mean_between / mean_within
    """
    populations = pop_labels["population"].unique()
    times = rates_df.columns.astype(float).values

    # Precompute within-population mean rates and pair lists
    within_pairs = {}
    within_mean  = {}
    for pop in populations:
        samples = np.unique(pop_labels[pop_labels["population"] == pop][id_col].values)
        pairs = get_pairs(samples, samples, within=True)
        within_pairs[pop] = pairs
        if len(pairs) > 0:
            within_mean[pop] = rates_df.loc[pairs].values.mean(axis=0)
        else:
            within_mean[pop] = np.zeros(len(times))

    results = {}
    for popA, popB in combinations(populations, 2):
        samplesA = np.unique(pop_labels[pop_labels["population"] == popA][id_col].values)
        samplesB = np.unique(pop_labels[pop_labels["population"] == popB][id_col].values)

        between_pairs = get_pairs(samplesA, samplesB, within=False)
        if len(between_pairs) == 0:
            continue

        mean_between = rates_df.loc[between_pairs].values.mean(axis=0)

        if len(within_pairs[popA]) == 0 or len(within_pairs[popB]) == 0:
            mean_within = np.full(len(times), np.nan)
        else:
            mean_within = (within_mean[popA] + within_mean[popB]) / 2

        with np.errstate(divide="ignore", invalid="ignore"):
            rho = np.where(mean_within > 0, mean_between / mean_within, np.nan)

        results[(popA, popB)] = {
            "between_pairs": between_pairs,
            "mean_between":  mean_between,
            "mean_within":   mean_within,
            "rho":           rho,
        }

    return results


def plot_relative_divergence(rates_df, pop_colors, pop_labels,
                              min_time=0, time_scale="linear", id_col="sample_id"):
    setup_plotting_style()

    pop_colors_dict = dict(zip(pop_colors["population"], pop_colors["color"]))
    times = rates_df.columns.astype(float).values
    populations = pop_labels["population"].unique()

    results = compute_relative_divergence(rates_df, pop_labels, id_col=id_col)
    pairs = list(results.keys())
    n_pairs = len(pairs)
    if n_pairs == 0:
        print("Warning: No population pairs found. Skipping relative divergence plot.")
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.text(0.5, 0.5, "No between-population pairs available\n(insufficient within-population samples)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_visible(True)
        return fig, fig

    # One panel per pair + one summary panel at the end
    n_panels = n_pairs + 1
    ncols = min(n_panels, 4)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    time_mask = times >= min_time

    # Build a large non-repeating color+linestyle palette for the summary panel
    tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    linestyles = ["-", "--", "-.", ":"]
    pair_styles = {}
    for i, pair in enumerate(pairs):
        pair_styles[pair] = {
            "color":     tab20_colors[i % 20],
            "linestyle": linestyles[(i // 20) % len(linestyles)],
        }

    # Compute y limits from mean lines only, with log-scale padding factor of 2
    all_rho_means = np.concatenate([res["rho"][time_mask] for res in results.values()])
    finite = all_rho_means[np.isfinite(all_rho_means) & (all_rho_means > 0)]
    ymin = finite.min() / 2
    ymax = finite.max() * 2

    for idx, (popA, popB) in enumerate(pairs):
        ax = axes_flat[idx]
        res = results[(popA, popB)]
        color = pop_colors_dict.get(popB, "steelblue")

        # Individual between-population pair traces
        for pair in res["between_pairs"]:
            rho_pair = rates_df.loc[pair].values / res["mean_within"]
            ax.plot(times[time_mask], rho_pair[time_mask],
                    color=color, lw=INDIVIDUAL_LINE_WIDTH, alpha=INDIVIDUAL_LINE_ALPHA)

        # Mean rho (black)
        ax.plot(times[time_mask], res["rho"][time_mask],
                color="black", lw=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA)

        # Reference line at rho = 1
        ax.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)

        ax.set_title(f"{popA} vs {popB}")
        ax.set_xscale(time_scale)
        ax.set_yscale("log")
        ax.set_xlim(float(min_time), times.max())
        ax.set_ylim(ymin, ymax)

    # Summary panel: all pairs as mean lines with distinct color+linestyle
    ax_summary = axes_flat[n_pairs]
    summary_handles = []
    for popA, popB in pairs:
        res = results[(popA, popB)]
        s = pair_styles[(popA, popB)]
        ax_summary.plot(times[time_mask], res["rho"][time_mask],
                        color=s["color"], linestyle=s["linestyle"],
                        lw=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA)
        summary_handles.append(Line2D([0], [0], color=s["color"],
                                      linestyle=s["linestyle"], linewidth=2,
                                      label=f"{popA} vs {popB}"))
    ax_summary.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)
    ax_summary.set_title("All pairs")
    ax_summary.set_xscale(time_scale)
    ax_summary.set_yscale("log")
    ax_summary.set_xlim(float(min_time), times.max())
    ax_summary.set_ylim(ymin, ymax)
    ax_summary.legend(handles=summary_handles, title="Pair",
                      bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

    for ax in axes_flat[:n_panels]:
        ax.set_xlabel("Time (generations ago)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\rho_{AB}$ (between / within)")

    fig.tight_layout()

    # Standalone summary figure
    fig_summary, ax_s = plt.subplots(figsize=(6, 4))
    for popA, popB in pairs:
        res = results[(popA, popB)]
        s = pair_styles[(popA, popB)]
        ax_s.plot(times[time_mask], res["rho"][time_mask],
                  color=s["color"], linestyle=s["linestyle"],
                  lw=MEAN_LINE_WIDTH, alpha=MEAN_LINE_ALPHA,
                  label=f"{popA} vs {popB}")
    ax_s.axhline(1.0, color="black", lw=1, linestyle="--", alpha=0.5)
    ax_s.set_xscale(time_scale)
    ax_s.set_yscale("log")
    ax_s.set_xlim(float(min_time), times.max())
    ax_s.set_ylim(ymin, ymax)
    ax_s.set_xlabel("Time (generations ago)")
    ax_s.set_ylabel(r"$\rho_{AB}$ (between / within)")
    ax_s.legend(title="Pair", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig_summary.tight_layout()

    return fig, fig_summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot between/within coalescence rate ratio per population pair"
    )
    parser.add_argument("--input",        required=True, help="Path to rates parquet file")
    parser.add_argument("--colors",       required=True, help="Path to population colors CSV")
    parser.add_argument("--labels",       required=True, help="Path to population labels CSV")
    parser.add_argument("--output_path",  required=True, help="Output directory")
    parser.add_argument("--output_name",  required=True, help="Output filename (PNG)")
    parser.add_argument("--min_time",     type=float, default=0, help="Minimum time to plot")
    parser.add_argument("--time_scale",   type=str, default="linear",
                        help="Time scale: linear or log")
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true",
                        help="If set, input data is treated as per-haplotype (sample_id). Otherwise per-individual (individual_id)")

    args = parser.parse_args()
    per_haplotype = args.per_haplotype

    rates_df = pd.read_parquet(args.input)
    rates_df.columns = rates_df.columns.astype(float)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)

    if per_haplotype:
        id_col = "sample_id"
    else:
        id_col = "individual_id"
        if "individual_id" not in labels_df.columns:
            labels_df = labels_df.copy()
            labels_df["individual_id"] = labels_df["sample_id"] // 2

    active_pairs  = rates_df.index.values
    active_samples = np.unique([s for pair in active_pairs for s in pair])
    pop_labels = labels_df[labels_df[id_col].isin(active_samples)]
    populations = pop_labels["population"].unique()
    pop_colors  = colors_df[colors_df["population"].isin(populations)]

    print(f"Plotting relative divergence for {args.input}")

    fig, fig_summary = plot_relative_divergence(
        rates_df=rates_df,
        pop_colors=pop_colors,
        pop_labels=pop_labels,
        min_time=args.min_time,
        time_scale=args.time_scale,
        id_col=id_col,
    )
    out = os.path.join(args.output_path, args.output_name)
    summary_out = os.path.join(args.output_path,
                               args.output_name.replace(".png", "_summary.png"))
    fig.savefig(out, dpi=500)
    fig_summary.savefig(summary_out, dpi=500)
    print(f"Saved to {out}")
    print(f"Saved summary to {summary_out}")


if __name__ == "__main__":
    main()
