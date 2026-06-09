#%%
#!/usr/bin/env python3
import math
import numpy as np
import pandas as pd
import sys
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from plot_rates import plot_rates_multi_figs, plot_rates_single_fig

sys.path.append("../../src")
sys.path.append("../src")
from optimization_functions import compute_f
from plotting_style import *
import argparse
import matplotlib.pyplot as plt
import os

def compute_distance_df(embeddings, id_label="sample_id"):
    x_cols = sorted(c for c in embeddings.columns if c.startswith("X"))

    # Pivot each dimension to (n_individuals, n_times), then stack to
    # (n_individuals, n_times, n_dims). Avoids the O(n²·T) cross-join.
    pivots = [
        embeddings.pivot_table(index=id_label, columns="time", values=c, aggfunc="first")
        for c in x_cols
    ]
    individuals = pivots[0].index.values
    times       = pivots[0].columns.values
    X = np.stack([p.values for p in pivots], axis=-1)  # (n, n_times, n_dims)

    ii, jj   = np.triu_indices(len(individuals), k=1)
    distances = np.linalg.norm(X[ii] - X[jj], axis=-1)  # (n_pairs, n_times)

    pair_index = pd.MultiIndex.from_arrays(
        [individuals[ii], individuals[jj]],
        names=[f"{id_label}_1", f"{id_label}_2"],
    )
    return pd.DataFrame(distances, index=pair_index, columns=times)

    return D_df

def reconstruct_rates(D_df, N_min, k=1, link="inverse_distance"):
    reconstructed_rates,_ = compute_f(D_df, N_min, k=k, link=link)
    return reconstructed_rates

def reconstructed_vs_input_all_times(input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label="sample_id", ax=None):
    # When an ax is supplied, draw onto it (for embedding in a grid); otherwise make a standalone figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    input_rates = input_rates.copy()
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates = reconstructed_rates.copy()
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    pairs = input_rates.index
    n_times = len(input_rates.columns)

    pop_map = pop_labels.drop_duplicates(subset=id_label).set_index(id_label)["population"]
    color_map = pop_colors.set_index("population")["color"]

    pop1 = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2 = np.array([pop_map.get(p[1], "unknown") for p in pairs])

    # Flatten to (n_pairs * n_times,) — one entry per (pair, time) observation
    x_flat = input_rates.values.ravel()
    y_flat = reconstructed_rates.values.ravel()
    pop1_flat = np.repeat(pop1, n_times)
    pop2_flat = np.repeat(pop2, n_times)

    # Per-point fill/border colors: fill = sorted-first-pop, border = sorted-second-pop
    facecolors, edgecolors, linewidths = [], [], []
    for p1, p2 in zip(pop1_flat, pop2_flat):
        key = sorted([p1, p2])
        ca = color_map.get(key[0], "#808080")
        cb = color_map.get(key[1], "#808080")
        facecolors.append(ca)
        edgecolors.append(cb)
        linewidths.append(0.0 if p1 == p2 else 0.6)

    valid = np.isfinite(x_flat) & np.isfinite(y_flat) & (x_flat > 0) & (y_flat > 0)
    x_v  = x_flat[valid]
    y_v  = y_flat[valid]
    fc_v = np.array(facecolors)[valid]
    ec_v = np.array(edgecolors)[valid]
    lw_v = np.array(linewidths)[valid]

    rng   = np.random.default_rng(seed=0)
    order = rng.permutation(len(x_v))

    lims = [min(x_v.min(), y_v.min()), max(x_v.max(), y_v.max())]
    ax.plot(lims, lims, 'k--', lw=1.5, zorder=0)
    ax.scatter(x_v[order], y_v[order], facecolors=fc_v[order], edgecolors=ec_v[order],
               linewidths=lw_v[order], alpha=0.15, s=2, zorder=1)

    # Median diamonds — collect all groups, shuffle, single scatter to avoid biased overprinting
    unique_keys = list(dict.fromkeys(tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)))
    med_xs, med_ys, med_fcs, med_ecs, med_lws, legend_handles = [], [], [], [], [], []
    for key in unique_keys:
        mask  = np.array([tuple(sorted([p1, p2])) == key for p1, p2 in zip(pop1, pop2)])
        med_x = np.nanmedian(input_rates.values[mask], axis=0)
        med_y = np.nanmedian(reconstructed_rates.values[mask], axis=0)
        vt    = np.isfinite(med_x) & np.isfinite(med_y) & (med_x > 0) & (med_y > 0)
        ca    = color_map.get(key[0], "#808080")
        cb    = color_map.get(key[1], "#808080")
        lw    = 0.0 if key[0] == key[1] else 1.0
        label = key[0] if key[0] == key[1] else f"{key[0]}–{key[1]}"
        n = int(vt.sum())
        med_xs.extend(med_x[vt]);  med_ys.extend(med_y[vt])
        med_fcs.extend([ca] * n);  med_ecs.extend([cb] * n);  med_lws.extend([lw] * n)
        legend_handles.append(Line2D([0], [0], marker='D', linestyle='none',
                                     markerfacecolor=ca, markeredgecolor=cb,
                                     markeredgewidth=lw, markersize=5, label=label))

    med_order = np.random.default_rng(seed=1).permutation(len(med_xs))
    ax.scatter(np.array(med_xs)[med_order], np.array(med_ys)[med_order],
               facecolors=np.array(med_fcs)[med_order], edgecolors=np.array(med_ecs)[med_order],
               linewidths=np.array(med_lws)[med_order], s=30, marker='D', zorder=5)

    ax.set_xlabel("input rates + 1/N_max")
    ax.set_ylabel("reconstructed rates")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=7, markerscale=1.5,
              handlelength=1.2, borderpad=0.5, loc="lower right")

    r2     = np.corrcoef(x_v, y_v)[0, 1] ** 2
    log_r2 = np.corrcoef(np.log(x_v), np.log(y_v))[0, 1] ** 2
    corr, _ = spearmanr(x_v, y_v)
    ax.text(0.05, 0.95, f"$r^2$ = {r2:.3f}",           transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.05, 0.90, f"$r^2$ (log) = {log_r2:.3f}", transform=ax.transAxes, fontsize=8, va='top')
    ax.text(0.05, 0.85, f"Spearman $\\rho$ = {corr:.3f}", transform=ax.transAxes, fontsize=8, va='top')

    return fig

def reconstructed_vs_input_query(sample_id, input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label="sample_id"):
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    assert input_rates.shape == reconstructed_rates.shape, "Input and reconstructed rates must have the same shape"
    assert np.all(input_rates.columns.values == reconstructed_rates.columns.values), "Input and reconstructed rates must have the same time points"
    assert input_rates.index.equals(reconstructed_rates.index), "Input and reconstructed rates must have the same samples"

    pairs = input_rates.index.values
    pair_mask = np.array([(sample_id in pair) for pair in pairs])
    pair_indices = np.where(pair_mask)[0]
    other_sample_ids = np.array([pair[0] if pair[1] == sample_id else pair[1] for pair in pairs[pair_mask]])
    other_sample_labels = pop_labels.drop_duplicates(subset=id_label).set_index(id_label).loc[other_sample_ids]["population"].values
    other_sample_colors = pop_colors.set_index("population").loc[other_sample_labels]["color"].values

    filtered_input_rates = input_rates.iloc[pair_indices].values
    filtered_reconstructed_rates = reconstructed_rates.iloc[pair_indices].values

    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(filtered_input_rates.shape[0]):
        ax.scatter(
            filtered_input_rates[i, :],
            filtered_reconstructed_rates[i, :],
            alpha=0.1,
            s=1,
            color=other_sample_colors[i]
        )
    ax.set_xlabel("input rates + 1/N_max")
    ax.set_ylabel("reconstructed rates")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    # y=x line for reference
    ax.plot([np.min(filtered_input_rates), np.max(filtered_input_rates)], [np.min(filtered_input_rates), np.max(filtered_input_rates)], 'k--', lw=2)
    ## Add r^2 value to the plot
    input_rate_values = filtered_input_rates.flatten()
    reconstructed_rate_values = filtered_reconstructed_rates.flatten()
    r_squared = np.corrcoef(input_rate_values, reconstructed_rate_values)[0, 1] ** 2
    corr, p_value = spearmanr(input_rate_values, reconstructed_rate_values)
    ax.text(0.05, 0.95, f"$r^2$ = {r_squared:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, f"Spearman $\\rho$ = {corr:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top')

    return fig

def reconstructed_vs_input_interactive(input_rates, reconstructed_rates, pop_labels, pop_colors, N_max, top_k=2000, min_pairs_per_slice=10, id_label="sample_id"):
    input_rates = input_rates.copy()
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates = reconstructed_rates.copy()
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    pairs = input_rates.index
    times = input_rates.columns.values

    pop_map   = pop_labels.drop_duplicates(subset=id_label).set_index(id_label)["population"]
    color_map = pop_colors.set_index("population")["color"]

    pop1      = np.array([pop_map.get(p[0], "unknown") for p in pairs])
    pop2      = np.array([pop_map.get(p[1], "unknown") for p in pairs])
    pair_keys = [tuple(sorted([p1, p2])) for p1, p2 in zip(pop1, pop2)]
    unique_keys = list(dict.fromkeys(pair_keys))

    # Precompute medians from the full dataset per pair type and time
    medians = {}
    for key in unique_keys:
        mask = np.array([k == key for k in pair_keys])
        medians[key] = (
            np.nanmedian(input_rates.values[mask], axis=0),
            np.nanmedian(reconstructed_rates.values[mask], axis=0),
        )

    # Z-score within each time slice; skip slices with too few valid pairs
    log_ratio = np.abs(np.log(reconstructed_rates.values / input_rates.values))
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

    # Select top K by z-score
    flat_z = z_scores.copy()
    flat_z[~np.isfinite(flat_z)] = -np.inf
    flat_indices = np.argsort(flat_z, axis=None)[-top_k:][::-1]
    pair_indices, time_indices = np.unravel_index(flat_indices, z_scores.shape)

    # Capture full-data axis limits
    all_vals = np.concatenate([input_rates.values.ravel(), reconstructed_rates.values.ravel()])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    rate_min = float(all_vals.min())
    rate_max = float(all_vals.max())

    df = pd.DataFrame({
        "sample_1":           [pairs[i][0] for i in pair_indices],
        "sample_2":           [pairs[i][1] for i in pair_indices],
        "time":               times[time_indices],
        "input_rate":         input_rates.values[pair_indices, time_indices],
        "reconstructed_rate": reconstructed_rates.values[pair_indices, time_indices],
        "z_score":            z_scores[pair_indices, time_indices],
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

    # Randomize trace draw order to avoid systematic population burial
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
            x=sub["input_rate"], y=sub["reconstructed_rate"],
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
                "Input rate: %{x:.3e}<br>"
                "Reconstructed rate: %{y:.3e}<br>"
                "Z-score: %{customdata[5]:.2f}"
                "<extra></extra>"
            ),
            name=label, legendgroup=label,
        ))

        # Median diamonds computed from all pairs (not just top-K outliers)
        med_x, med_y = medians[key]
        vt = np.isfinite(med_x) & np.isfinite(med_y) & (med_x > 0) & (med_y > 0)
        if vt.sum() > 0:
            fig.add_trace(go.Scatter(
                x=med_x[vt], y=med_y[vt],
                mode="markers",
                marker=dict(
                    size=10, symbol="diamond", color=ca, opacity=1.0,
                    line=dict(color=cb, width=lw + 0.5 if lw > 0 else 0),
                ),
                customdata=times[vt].reshape(-1, 1),
                name=f"{label} (median)", legendgroup=label,
                hovertemplate=f"<b>{label} median</b><br>Time: %{{customdata[0]:.1f}}<br>Input: %{{x:.3e}}<br>Reconstructed: %{{y:.3e}}<extra></extra>",
            ))

    fig.update_layout(
        xaxis=dict(title="Input rate + 1/N_max", type="log",
                   range=[math.log10(rate_min), math.log10(rate_max)], exponentformat="e"),
        yaxis=dict(title="Reconstructed rate",   type="log",
                   range=[math.log10(rate_min), math.log10(rate_max)], exponentformat="e"),
        title=f"Top-{top_k} outlier (pair × time) points by within-timepoint z-score",
    )
    return fig


def find_percent_error(input_rates, reconstructed_rates, N_max):
    input_rates += 1/N_max
    error = np.abs(reconstructed_rates - input_rates) / np.abs(input_rates)

    return error
    
def main():
    argparser = argparse.ArgumentParser(description="Evaluate performance of rate reconstruction from embeddings")
    argparser.add_argument("--embeddings", type=str, default=None, help="Path to embeddings parquet file (not needed if --reconstructed_rates provided)")
    argparser.add_argument("--rates", type=str, required=True, help="Path to true rates parquet file")
    argparser.add_argument("--reconstructed_rates", type=str, default=None, help="Path to precomputed reconstructed rates parquet (skips recomputation)")
    argparser.add_argument("--colors", type=str, required=True, help="Path to population colors csv file")
    argparser.add_argument("--labels", type=str, required=True, help="Path to population labels csv file")
    argparser.add_argument("--N_min", type=float, default=None, help="N_min used to compute embeddings")
    argparser.add_argument("--N_max", type=float, default=None, help="N_max used to compute embeddings")
    argparser.add_argument("--k", type=int, default=1, help="Link function exponent used to compute embeddings")
    argparser.add_argument("--link", type=str, default="inverse_distance", help="Link function used to compute embeddings")
    argparser.add_argument("--data_output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--plot_output_path", type=str, required=True, help="Path to output directory")
    argparser.add_argument("--data_output_name", type=str, required=True, help="Name of the output file")
    argparser.add_argument("--plot_output_name", type=str, required=True, help="Name of the output file")
    argparser.add_argument("--min_time", type=float, default=0, help="Minimum time for plotting")
    argparser.add_argument("--max_rate_plot", type=float, default=None, help="Maximum rate for plotting")
    argparser.add_argument("--time_scale", default="linear", help="Time scale for demography plot, linear or log", type=str)
    argparser.add_argument("--phased", type=str, default="False", help="Indicates whether the data is phased. If False, rates/embeddings are per-individual (individual_id) rather than per-haplotype (sample_id)")
    args = argparser.parse_args()

    # Loading data
    phased = args.phased == "True"
    id_label = "sample_id" if phased else "individual_id"
    N_min = args.N_min
    N_max = args.N_max
    k = args.k
    link = args.link
    rates = pd.read_parquet(args.rates)
    rates.columns = rates.columns.astype(float)
    pop_colors = pd.read_csv(args.colors)
    pop_labels = pd.read_csv(args.labels)

    if args.reconstructed_rates is not None:
        reconstructed_rates = pd.read_parquet(args.reconstructed_rates)
        reconstructed_rates.columns = reconstructed_rates.columns.astype(float)
        print(f"Loaded precomputed reconstructed rates from {args.reconstructed_rates}")
    else:
        if args.embeddings is None:
            raise ValueError("Either --embeddings or --reconstructed_rates must be provided")
        embeddings = pd.read_parquet(args.embeddings)
        D_df = compute_distance_df(embeddings, id_label=id_label)
        reconstructed_rates = reconstruct_rates(D_df, N_min=N_min, k=k, link=link)
        reconstructed_rates.to_parquet(f"{args.data_output_path}/{args.data_output_name}", index=True)

    # Align to common timepoints (embeddings may have been subsampled)
    common_times = sorted(set(rates.columns) & set(reconstructed_rates.columns))
    rates = rates[common_times]
    reconstructed_rates = reconstructed_rates[common_times]

    # Align to common pairs (individual-level rates and reconstructed may differ)
    common_pairs = rates.index.intersection(reconstructed_rates.index)
    if len(common_pairs) < len(rates):
        print(f"  [warn] {len(rates) - len(common_pairs)} pairs in rates not in reconstructed_rates — dropping")
    if len(common_pairs) < len(reconstructed_rates):
        print(f"  [warn] {len(reconstructed_rates) - len(common_pairs)} pairs in reconstructed_rates not in rates — dropping")
    rates = rates.loc[common_pairs]
    reconstructed_rates = reconstructed_rates.loc[common_pairs]

    print(f"Evaluating performance: {args.rates}")

    fig_interactive = reconstructed_vs_input_interactive(rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label=id_label)
    interactive_output_name = args.plot_output_name.replace(".png", ".html")
    fig_interactive.write_html(f"{args.plot_output_path}/{interactive_output_name}")

    # True vs Reconstructed rates across all time points
    tvr_fig = reconstructed_vs_input_all_times(rates, reconstructed_rates, pop_labels, pop_colors, N_max, id_label=id_label)
    tvr_fig.savefig(f"{args.plot_output_path}/{args.plot_output_name}", dpi=500, bbox_inches="tight")

    # Percent error in reconstructed rates
    # error = find_percent_error(rates, reconstructed_rates, N_max)
    # error.to_parquet(f"{args.data_output_path}/{args.data_output_name}", index=True)

    # Plotting reconstructed rates over time
    if len(pop_labels["population"].unique()) < 10: # only plot if there are less than 10 populations for better visualization
        rates_fig = plot_rates_multi_figs(reconstructed_rates, pop_colors, pop_labels, min_time=args.min_time, max_rate_plot=args.max_rate_plot, time_scale=args.time_scale, ylabel="Coalescence Rate", phased=phased)
        rates_fig_single = plot_rates_single_fig(reconstructed_rates, min_time = args.min_time, max_rate_plot = args.max_rate_plot, time_scale=args.time_scale, ylabel="Coalescence Rate")

        rates_fig.savefig(os.path.join(args.plot_output_path, args.plot_output_name.replace("performance", "reconstructed_rates")), dpi=500, bbox_inches="tight")
        rates_fig_single.savefig(os.path.join(args.plot_output_path, args.plot_output_name.replace("performance", "reconstructed_rates_single")), dpi=500, bbox_inches="tight")

if __name__ == "__main__":
    main()
