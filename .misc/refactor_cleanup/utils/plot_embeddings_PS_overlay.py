#!/usr/bin/env python3
"""Single 3-D overlay plot of all SINGER posterior sample embeddings.

For each individual:
  - N thin, semi-transparent trajectories (one per PS, Procrustes-aligned to PS0)
    coloured with a slight lightness variation per individual within each population
  - 1 thick trajectory = posterior mean X1/X2 across all PS

Exports HTML (interactive) and PNG (static).
"""
import argparse
import colorsys
import glob
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.linalg import svd

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from PlotEmbeddings import PlotEmbeddings


def _discover_embeddings(ps_data_dir):
    matches = sorted(
        m for m in glob.glob(os.path.join(ps_data_dir, "embeddings.*.parquet"))
        if "loss" not in os.path.basename(m)
    )
    if not matches:
        raise FileNotFoundError(f"No embeddings*.parquet in {ps_data_dir}")
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime)
        print(f"  [warn] multiple embeddings in {ps_data_dir}; using newest")
    return matches[-1]


def _procrustes_align(dfs, id_col):
    most_recent = min(df["time"].min() for df in dfs)
    ref_slice   = dfs[0][dfs[0]["time"] == most_recent].set_index(id_col)
    common_ids  = sorted(
        set.intersection(*[set(df[df["time"] == most_recent][id_col]) for df in dfs])
    )
    ref_xy = ref_slice.loc[common_ids, ["X1", "X2"]].values
    aligned = [dfs[0]]
    for df in dfs[1:]:
        s        = df[df["time"] == most_recent].set_index(id_col)
        src_xy   = s.loc[common_ids, ["X1", "X2"]].values
        src_mean = src_xy.mean(axis=0)
        ref_mean = ref_xy.mean(axis=0)
        U, _, Vt = svd((src_xy - src_mean).T @ (ref_xy - ref_mean))
        R        = U @ Vt
        df2      = df.copy()
        rotated  = (df2[["X1", "X2"]].values - src_mean) @ R + ref_mean
        df2["X1"] = rotated[:, 0]
        df2["X2"] = rotated[:, 1]
        aligned.append(df2)
    return aligned, most_recent


def _compute_mean_trajectories(dfs, id_col):
    combined = pd.concat(
        [df[[id_col, "time", "X1", "X2"]] for df in dfs],
        ignore_index=True,
    )
    return combined.groupby([id_col, "time"], as_index=False)[["X1", "X2"]].mean()


def _hex_to_hls(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _individual_color_map(individuals_by_pop, base_color_map, l_spread=0.18):
    """
    Assign each individual a unique lightness variant of its population color.
    Individuals are sorted by id so the mapping is deterministic.
    """
    result = {}
    for pop, inds in individuals_by_pop.items():
        base = base_color_map.get(pop, "#808080")
        h, l, s = _hex_to_hls(base)
        inds_sorted = sorted(inds)
        n = len(inds_sorted)
        l_values = (np.linspace(l - l_spread, l + l_spread, n)
                    if n > 1 else np.array([l]))
        for ind_id, li in zip(inds_sorted, l_values):
            result[ind_id] = _hls_to_hex(h, li, s)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Overlay all PS embeddings in one 3-D plot with posterior means"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_dir")
    group.add_argument("--inputs", nargs="+")

    parser.add_argument("--n_ps",        type=int,   default=8)
    parser.add_argument("--colors",      required=True)
    parser.add_argument("--labels",      required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", default="embeddings_PS_overlay")
    parser.add_argument("--min_time",    type=float, default=1000)
    parser.add_argument("--max_time",    type=float, default=100000)
    parser.add_argument("--time_scale",  default="log")
    parser.add_argument("--lims",        type=str,   default="None")
    parser.add_argument("--thin_width",  type=float, default=0.8)
    parser.add_argument("--thin_alpha",  type=float, default=0.25)
    parser.add_argument("--mean_width",  type=float, default=3.0)
    parser.add_argument("--mean_alpha",  type=float, default=1.0)
    parser.add_argument("--l_spread",    type=float, default=0.18,
                        help="Lightness range (±) for per-individual color tone")
    parser.add_argument("--eye_x",       type=float, default=1.8)
    parser.add_argument("--eye_y",       type=float, default=1.8)
    parser.add_argument("--eye_z",       type=float, default=0.3)
    args = parser.parse_args()

    lims = None if args.lims == "None" else float(args.lims)

    if args.inputs:
        input_paths = args.inputs
        ps_indices  = list(range(len(input_paths)))
    else:
        ps_indices  = list(range(args.n_ps))
        input_paths = []
        for k in ps_indices:
            p = _discover_embeddings(os.path.join(args.base_dir, f"chimps_PS{k}", "data"))
            input_paths.append(p)
            print(f"PS{k}: {os.path.basename(p)}")

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)

    print("Loading embeddings...")
    dfs = []
    for path in input_paths:
        df = pd.read_parquet(path)
        df = df[(df["time"] >= args.min_time) & (df["time"] <= args.max_time)]
        dfs.append(df)

    id_col = "individual_id" if "individual_id" in dfs[0].columns else "sample_id"
    if id_col == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2

    pop_map       = labels_df.drop_duplicates(subset=id_col).set_index(id_col)["population"]
    pop_color_map = colors_df.set_index("population")["color"].to_dict()

    # Per-individual color (slight lightness variation within population)
    all_ids = set(dfs[0][id_col].unique())
    individuals_by_pop = {}
    for ind_id in all_ids:
        pop = pop_map.get(ind_id, "unknown")
        individuals_by_pop.setdefault(pop, set()).add(ind_id)
    ind_color_map = _individual_color_map(individuals_by_pop, pop_color_map, args.l_spread)

    print("Procrustes-aligning...")
    dfs, _ = _procrustes_align(dfs, id_col)

    print("Computing posterior means...")
    mean_df = _compute_mean_trajectories(dfs, id_col)
    mean_df["population"] = mean_df[id_col].map(pop_map).fillna("unknown")

    if lims is None:
        all_x = np.concatenate([df["X1"].dropna().values for df in dfs])
        all_y = np.concatenate([df["X2"].dropna().values for df in dfs])
        cx    = (all_x.max() + all_x.min()) / 2
        cy    = (all_y.max() + all_y.min()) / 2
        half  = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.06
        xy_lims = max(abs(cx) + half, abs(cy) + half, half)
    else:
        xy_lims = lims

    plotter = PlotEmbeddings(
        embeddings_df=dfs[0].copy(),
        labels_df=labels_df,
        colors_df=colors_df,
        sample_id_col=id_col,
    )
    plotter.init_plot(
        time_scale=args.time_scale,
        xy_lims=xy_lims,
        min_time=args.min_time,
        max_time=args.max_time,
    )
    fig = plotter.fig

    def _tplot(t_series):
        return np.maximum(t_series.values.astype(float), plotter.time_eps_plot)

    # ── Thin PS lines — individual-toned color ───────────────────────────────
    print("Adding thin PS trajectories...")
    for k, (df, ps_idx) in enumerate(zip(dfs, ps_indices)):
        for ind_id in df[id_col].unique():
            row = df[df[id_col] == ind_id].sort_values("time")
            if row.empty:
                continue
            pop   = pop_map.get(ind_id, "unknown")
            color = ind_color_map.get(ind_id, pop_color_map.get(pop, "#808080"))
            fig.add_trace(go.Scatter3d(
                x=row["X1"].values,
                y=row["X2"].values,
                z=_tplot(row["time"]),
                mode="lines",
                line=dict(color=color, width=args.thin_width),
                opacity=args.thin_alpha,
                name=f"{pop} PS{ps_idx}",
                hovertemplate=f"{pop} {ind_id} PS{ps_idx}<br>t=%{{z:.0f}}<extra></extra>",
                showlegend=False,
                legendgroup=pop,
            ))

    # ── Thick mean lines — population base color ─────────────────────────────
    print("Adding mean trajectories...")
    populations_seen = set()
    for ind_id in mean_df[id_col].unique():
        row = mean_df[mean_df[id_col] == ind_id].sort_values("time")
        if row.empty:
            continue
        pop   = pop_map.get(ind_id, "unknown")
        color = pop_color_map.get(pop, "#808080")
        show_legend = pop not in populations_seen
        populations_seen.add(pop)
        fig.add_trace(go.Scatter3d(
            x=row["X1"].values,
            y=row["X2"].values,
            z=_tplot(row["time"]),
            mode="lines",
            line=dict(color=color, width=args.mean_width),
            opacity=args.mean_alpha,
            name=pop,
            hovertemplate=f"{pop} {ind_id} (mean)<br>t=%{{z:.0f}}<extra></extra>",
            showlegend=show_legend,
            legendgroup=pop,
        ))

    os.makedirs(args.output_path, exist_ok=True)
    out_base = os.path.join(args.output_path, args.output_name)

    fig.update_layout(
        legend=dict(title="Population", itemsizing="constant"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.write_html(f"{out_base}.html")
    print(f"Saved HTML: {out_base}.html")

    img, _ = plotter.save_static(
        width=900,
        height=800,
        scale=2,
        eye=dict(x=args.eye_x, y=args.eye_y, z=args.eye_z),
    )
    img.save(f"{out_base}.png")
    print(f"Saved PNG:  {out_base}.png")


if __name__ == "__main__":
    main()
