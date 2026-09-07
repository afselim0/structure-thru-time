#!/usr/bin/env python3
"""Single 3-D overlay plot across all PS runs, with one PS as the reference.

Reference PS: thick opaque trajectories in the canonical population color.
Other PSs:    thin semi-transparent trajectories, Procrustes-aligned to reference.

Exports HTML (interactive) and PNG (static).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    return matches[-1]


def _procrustes_align(dfs, id_col, ref_pos):
    most_recent = min(df["time"].min() for df in dfs)
    ref_slice   = dfs[ref_pos][dfs[ref_pos]["time"] == most_recent].set_index(id_col)
    common_ids  = sorted(
        set.intersection(*[set(df[df["time"] == most_recent][id_col]) for df in dfs])
    )
    ref_xy  = ref_slice.loc[common_ids, ["X1", "X2"]].values
    aligned = []
    for i, df in enumerate(dfs):
        if i == ref_pos:
            aligned.append(df)
            continue
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
    return aligned


def main():
    parser = argparse.ArgumentParser(
        description="3-D overlay of all PS runs with one reference PS highlighted"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_dir")
    group.add_argument("--inputs", nargs="+")

    parser.add_argument("--n_ps",         type=int,   default=8)
    parser.add_argument("--reference_ps", type=int,   required=True,
                        help="PS index to show thick; all others shown thin")
    parser.add_argument("--colors",       required=True)
    parser.add_argument("--labels",       required=True)
    parser.add_argument("--output_path",  required=True)
    parser.add_argument("--output_name",  default=None)
    parser.add_argument("--min_time",     type=float, default=1000)
    parser.add_argument("--max_time",     type=float, default=100000)
    parser.add_argument("--time_scale",   default="log")
    parser.add_argument("--lims",         type=str,   default="None")
    parser.add_argument("--ref_width",    type=float, default=3.0)
    parser.add_argument("--ref_alpha",    type=float, default=1.0)
    parser.add_argument("--thin_width",   type=float, default=0.8)
    parser.add_argument("--thin_alpha",   type=float, default=0.2)
    parser.add_argument("--eye_x",        type=float, default=1.8)
    parser.add_argument("--eye_y",        type=float, default=1.8)
    parser.add_argument("--eye_z",        type=float, default=0.3)
    args = parser.parse_args()

    if args.output_name is None:
        args.output_name = f"embeddings_PS{args.reference_ps}_reference"

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

    ref_pos = ps_indices.index(args.reference_ps) if args.reference_ps in ps_indices else 0
    print(f"Procrustes-aligning to PS{ps_indices[ref_pos]}...")
    dfs = _procrustes_align(dfs, id_col, ref_pos)

    if lims is None:
        all_x = np.concatenate([df["X1"].dropna().values for df in dfs])
        all_y = np.concatenate([df["X2"].dropna().values for df in dfs])
        cx    = (all_x.max() + all_x.min()) / 2
        cy    = (all_y.max() + all_y.min()) / 2
        half  = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.06
        xy_lims = max(abs(cx) + half, abs(cy) + half, half)
    else:
        xy_lims = lims

    # Use the reference PS df to initialise the figure axes
    plotter = PlotEmbeddings(
        embeddings_df=dfs[ref_pos].copy(),
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

    # ── Thin PSs first (drawn behind) ───────────────────────────────────────
    print("Adding thin PS trajectories...")
    for df, ps_idx in zip(dfs, ps_indices):
        if ps_idx == args.reference_ps:
            continue
        for ind_id in df[id_col].unique():
            row = df[df[id_col] == ind_id].sort_values("time")
            if row.empty:
                continue
            pop   = pop_map.get(ind_id, "unknown")
            color = pop_color_map.get(pop, "#808080")
            fig.add_trace(go.Scatter3d(
                x=row["X1"].values,
                y=row["X2"].values,
                z=_tplot(row["time"]),
                mode="lines",
                line=dict(color=color, width=args.thin_width),
                opacity=args.thin_alpha,
                hovertemplate=f"{pop} {ind_id} PS{ps_idx}<br>t=%{{z:.0f}}<extra></extra>",
                showlegend=False,
                legendgroup=pop,
            ))

    # ── Reference PS thick (drawn on top) ───────────────────────────────────
    print(f"Adding reference PS{args.reference_ps} trajectories...")
    populations_seen = set()
    ref_df = dfs[ref_pos]
    for ind_id in ref_df[id_col].unique():
        row = ref_df[ref_df[id_col] == ind_id].sort_values("time")
        if row.empty:
            continue
        pop         = pop_map.get(ind_id, "unknown")
        color       = pop_color_map.get(pop, "#808080")
        show_legend = pop not in populations_seen
        populations_seen.add(pop)
        fig.add_trace(go.Scatter3d(
            x=row["X1"].values,
            y=row["X2"].values,
            z=_tplot(row["time"]),
            mode="lines",
            line=dict(color=color, width=args.ref_width),
            opacity=args.ref_alpha,
            name=pop,
            hovertemplate=f"{pop} {ind_id} PS{args.reference_ps} [REF]<br>t=%{{z:.0f}}<extra></extra>",
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
