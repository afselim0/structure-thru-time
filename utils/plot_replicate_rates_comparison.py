#!/usr/bin/env python3
import argparse
import os
import sys

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
import plotly.graph_objects as go

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from evaluate_performance import compute_distance_df, reconstruct_rates


def compare_replicate_rates(rates_a, rates_b, pop_labels, pop_colors, label_a, label_b):
    """Scatter plot of reconstructed rates from replicate A (x) vs replicate B (y)."""
    rates_a = rates_a.copy()
    rates_b = rates_b.copy()
    rates_a.columns = rates_a.columns.astype(float)
    rates_b.columns = rates_b.columns.astype(float)

    # Align to common pairs and timepoints
    common_pairs = rates_a.index.intersection(rates_b.index)
    common_times = sorted(set(rates_a.columns) & set(rates_b.columns))
    rates_a = rates_a.loc[common_pairs, common_times]
    rates_b = rates_b.loc[common_pairs, common_times]

    pairs   = rates_a.index
    n_times = len(common_times)

    pop_map   = pop_labels.set_index("sample_id")["population"]
    color_map = pop_colors.set_index("population")["color"]

    pop1 = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2 = np.array([pop_map.get(p[1], "unknown") for p in pairs])

    x_flat    = rates_a.values.ravel()
    y_flat    = rates_b.values.ravel()
    pop1_flat = np.repeat(pop1, n_times)
    pop2_flat = np.repeat(pop2, n_times)

    facecolors, edgecolors, linewidths = [], [], []
    for p1, p2 in zip(pop1_flat, pop2_flat):
        key = sorted([p1, p2])
        ca  = color_map.get(key[0], "#808080")
        cb  = color_map.get(key[1], "#808080")
        facecolors.append(ca)
        edgecolors.append(cb)
        linewidths.append(0.0 if p1 == p2 else 0.6)

    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 0) & (y_flat > 0)
    x_v   = x_flat[valid]
    y_v   = y_flat[valid]
    fc_v  = np.array(facecolors)[valid]
    ec_v  = np.array(edgecolors)[valid]
    lw_v  = np.array(linewidths)[valid]

    rng   = np.random.default_rng(seed=0)
    order = rng.permutation(len(x_v))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax_lims = [min(x_v.min(), y_v.min()), max(x_v.max(), y_v.max())]
    ax.plot(ax_lims, ax_lims, 'k--', lw=1.5, zorder=0)
    ax.scatter(x_v[order], y_v[order], facecolors=fc_v[order], edgecolors=ec_v[order],
               linewidths=lw_v[order], alpha=0.15, s=2, zorder=1)

    # Median diamonds per population pair
    unique_keys  = list(dict.fromkeys(tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)))
    med_xs, med_ys, med_fcs, med_ecs, med_lws, legend_handles = [], [], [], [], [], []
    for key in unique_keys:
        mask  = np.array([tuple(sorted([p1, p2])) == key for p1, p2 in zip(pop1, pop2)])
        med_x = np.nanmedian(rates_a.values[mask], axis=0)
        med_y = np.nanmedian(rates_b.values[mask], axis=0)
        vt    = np.isfinite(med_x) & np.isfinite(med_y) & (med_x > 0) & (med_y > 0)
        ca    = color_map.get(key[0], "#808080")
        cb    = color_map.get(key[1], "#808080")
        lw    = 0.0 if key[0] == key[1] else 1.0
        label = key[0] if key[0] == key[1] else f"{key[0]}–{key[1]}"
        n     = int(vt.sum())
        med_xs.extend(med_x[vt]);  med_ys.extend(med_y[vt])
        med_fcs.extend([ca] * n);  med_ecs.extend([cb] * n);  med_lws.extend([lw] * n)
        legend_handles.append(Line2D([0], [0], marker='D', linestyle='none',
                                     markerfacecolor=ca, markeredgecolor=cb,
                                     markeredgewidth=lw, markersize=5, label=label))

    med_order = np.random.default_rng(seed=1).permutation(len(med_xs))
    ax.scatter(np.array(med_xs)[med_order], np.array(med_ys)[med_order],
               facecolors=np.array(med_fcs)[med_order],
               edgecolors=np.array(med_ecs)[med_order],
               linewidths=np.array(med_lws)[med_order],
               s=30, marker='D', zorder=5)

    ax.set_xlabel(f"reconstructed rates ({label_a})")
    ax.set_ylabel(f"reconstructed rates ({label_b})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=7, markerscale=1.5,
              handlelength=1.2, borderpad=0.5, loc="lower right")

    r2     = np.corrcoef(x_v, y_v)[0, 1] ** 2
    log_r2 = np.corrcoef(np.log(x_v), np.log(y_v))[0, 1] ** 2
    corr, _ = spearmanr(x_v, y_v)
    ax.text(0.05, 0.95, f"$r^2$ = {r2:.3f}",              transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.05, 0.90, f"$r^2$ (log) = {log_r2:.3f}",    transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.05, 0.85, f"Spearman $\\rho$ = {corr:.3f}", transform=ax.transAxes, fontsize=8, va='top')

    return fig


def compare_replicate_rates_interactive(rates_a, rates_b, pop_labels, pop_colors,
                                        label_a, label_b, top_k=2000, min_pairs_per_slice=10):
    """Interactive Plotly scatter of reconstructed rates: replicate A (x) vs replicate B (y)."""
    rates_a = rates_a.copy()
    rates_b = rates_b.copy()
    rates_a.columns = rates_a.columns.astype(float)
    rates_b.columns = rates_b.columns.astype(float)

    common_pairs = rates_a.index.intersection(rates_b.index)
    common_times = sorted(set(rates_a.columns) & set(rates_b.columns))
    rates_a = rates_a.loc[common_pairs, common_times]
    rates_b = rates_b.loc[common_pairs, common_times]

    pairs     = rates_a.index
    times     = np.array(common_times, dtype=float)
    pop_map   = pop_labels.set_index("sample_id")["population"]
    color_map = pop_colors.set_index("population")["color"]

    pop1      = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2      = np.array([pop_map.get(p[1], "unknown") for p in pairs])
    pair_keys = [tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)]
    unique_keys = list(dict.fromkeys(pair_keys))

    # Medians over all pairs per population-pair type and time
    medians = {}
    for key in unique_keys:
        mask = np.array([k == key for k in pair_keys])
        medians[key] = (
            np.nanmedian(rates_a.values[mask], axis=0),
            np.nanmedian(rates_b.values[mask], axis=0),
        )

    # Z-score of |log(b/a)| within each time slice to identify discordant points
    log_ratio = np.abs(np.log(rates_b.values / rates_a.values))
    z_scores  = np.full_like(log_ratio, np.nan)
    for t_idx in range(len(times)):
        col   = log_ratio[:, t_idx]
        valid = np.isfinite(col)
        if valid.sum() < min_pairs_per_slice:
            continue
        std_t = col[valid].std()
        if std_t < 1e-12:
            continue
        z_scores[valid, t_idx] = (col[valid] - col[valid].mean()) / std_t

    # Select top-K most discordant (pair, time) points
    flat_z = z_scores.copy()
    flat_z[~np.isfinite(flat_z)] = -np.inf
    flat_indices = np.argsort(flat_z, axis=None)[-top_k:][::-1]
    pair_indices, time_indices = np.unravel_index(flat_indices, z_scores.shape)

    all_vals = np.concatenate([rates_a.values.ravel(), rates_b.values.ravel()])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    rate_min = float(all_vals.min())
    rate_max = float(all_vals.max())

    df = pd.DataFrame({
        "sample_1": [pairs[i][0] for i in pair_indices],
        "sample_2": [pairs[i][1] for i in pair_indices],
        "time":     times[time_indices],
        "rate_a":   rates_a.values[pair_indices, time_indices],
        "rate_b":   rates_b.values[pair_indices, time_indices],
        "z_score":  z_scores[pair_indices, time_indices],
    })
    df["pop_1"]    = df["sample_1"].map(pop_map).fillna("unknown")
    df["pop_2"]    = df["sample_2"].map(pop_map).fillna("unknown")
    df["pair_key"] = [tuple(sorted([p1, p2])) for p1, p2 in zip(df["pop_1"], df["pop_2"])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[rate_min, rate_max], y=[rate_min, rate_max],
        mode="lines", line=dict(color="black", dash="dash", width=2),
        name="y = x", showlegend=True,
    ))

    rng = np.random.default_rng(seed=0)
    shuffled_keys = [unique_keys[i] for i in rng.permutation(len(unique_keys))]

    for key in shuffled_keys:
        sub = df[df["pair_key"] == key].sample(frac=1, random_state=0)
        if len(sub) == 0:
            continue
        ca    = color_map.get(key[0], "#808080")
        cb    = color_map.get(key[1], "#808080")
        lw    = 1.5 if key[0] != key[1] else 0
        label = key[0] if key[0] == key[1] else f"{key[0]}–{key[1]}"

        fig.add_trace(go.Scatter(
            x=sub["rate_a"], y=sub["rate_b"],
            mode="markers",
            marker=dict(size=5, color=ca, opacity=0.7, line=dict(color=cb, width=lw)),
            customdata=np.stack([
                sub["sample_1"].astype(str), sub["sample_2"].astype(str),
                sub["pop_1"].astype(str),    sub["pop_2"].astype(str),
                sub["time"].values,          sub["z_score"].values,
            ], axis=1),
            hovertemplate=(
                "<b>%{customdata[2]} — %{customdata[3]}</b><br>"
                "Samples: %{customdata[0]} / %{customdata[1]}<br>"
                "Time: %{customdata[4]:.1f}<br>"
                f"{label_a}: %{{x:.3e}}<br>"
                f"{label_b}: %{{y:.3e}}<br>"
                "Z-score: %{customdata[5]:.2f}"
                "<extra></extra>"
            ),
            name=label, legendgroup=label,
        ))

        med_x, med_y = medians[key]
        vt = np.isfinite(med_x) & np.isfinite(med_y) & (med_x > 0) & (med_y > 0)
        if vt.sum() > 0:
            fig.add_trace(go.Scatter(
                x=med_x[vt], y=med_y[vt],
                mode="markers",
                marker=dict(size=10, symbol="diamond", color=ca, opacity=1.0,
                            line=dict(color=cb, width=lw + 0.5 if lw > 0 else 0)),
                customdata=times[vt].reshape(-1, 1),
                name=f"{label} (median)", legendgroup=label,
                hovertemplate=(
                    f"<b>{label} median</b><br>"
                    "Time: %{customdata[0]:.1f}<br>"
                    f"{label_a}: %{{x:.3e}}<br>"
                    f"{label_b}: %{{y:.3e}}<extra></extra>"
                ),
            ))

    fig.update_layout(
        xaxis=dict(title=f"reconstructed rates ({label_a})", type="log",
                   range=[math.log10(rate_min), math.log10(rate_max)], exponentformat="e"),
        yaxis=dict(title=f"reconstructed rates ({label_b})", type="log",
                   range=[math.log10(rate_min), math.log10(rate_max)], exponentformat="e"),
        title=f"Replicate rates: {label_a} vs {label_b} — top-{top_k} discordant (pair × time) points",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare reconstructed rates between two embedding replicates"
    )
    parser.add_argument("--input_a",  required=True, help="Embeddings parquet for replicate A")
    parser.add_argument("--input_b",  required=True, help="Embeddings parquet for replicate B")
    parser.add_argument("--seed_a",   required=True, help="Label for replicate A (e.g. 'seed 0')")
    parser.add_argument("--seed_b",   required=True, help="Label for replicate B (e.g. 'seed 1')")
    parser.add_argument("--colors",   required=True, help="Population colors CSV")
    parser.add_argument("--labels",   required=True, help="Population labels CSV")
    parser.add_argument("--N_min",    type=float, required=True)
    parser.add_argument("--N_max",    type=float, required=True)
    parser.add_argument("--link_function", type=str, default="inverse_distance")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--param_label", type=str, default=None)
    parser.add_argument("--output_name_html", type=str, default=None,
                        help="Output filename for interactive HTML. Defaults to output_name with .html suffix.")
    args = parser.parse_args()

    pop_colors = pd.read_csv(args.colors)
    pop_labels = pd.read_csv(args.labels)

    print(f"Loading embeddings for {args.seed_a}...")
    emb_a = pd.read_parquet(args.input_a)
    print(f"Loading embeddings for {args.seed_b}...")
    emb_b = pd.read_parquet(args.input_b)

    print("Computing reconstructed rates...")
    rates_a = reconstruct_rates(compute_distance_df(emb_a), args.N_min, args.link_function)
    rates_b = reconstruct_rates(compute_distance_df(emb_b), args.N_min, args.link_function)

    # Subset labels to samples present in the embeddings
    active_samples = set(emb_a["sample_id"].unique())
    pop_labels = pop_labels[pop_labels["sample_id"].isin(active_samples)]
    populations = pop_labels["population"].unique()
    pop_colors = pop_colors[pop_colors["population"].isin(populations)]

    fig = compare_replicate_rates(
        rates_a, rates_b, pop_labels, pop_colors,
        label_a=args.seed_a, label_b=args.seed_b,
    )

    if args.param_label:
        fig.text(0.5, -0.08, args.param_label, ha="center", fontsize=7, color="#555555",
                 transform=fig.transFigure)

    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved static plot to {out_path}")

    print("Building interactive HTML...")
    fig_html = compare_replicate_rates_interactive(
        rates_a, rates_b, pop_labels, pop_colors,
        label_a=args.seed_a, label_b=args.seed_b,
    )
    if args.param_label:
        fig_html.update_layout(
            annotations=[dict(
                text=args.param_label, showarrow=False,
                xref="paper", yref="paper", x=0.5, y=-0.12,
                xanchor="center", font=dict(size=9, color="#555555"),
            )]
        )
    html_name = (args.output_name_html
                 if args.output_name_html
                 else args.output_name.replace(".png", ".html"))
    html_path = os.path.join(args.output_path, html_name)
    fig_html.write_html(html_path)
    print(f"Saved interactive plot to {html_path}")


if __name__ == "__main__":
    main()
