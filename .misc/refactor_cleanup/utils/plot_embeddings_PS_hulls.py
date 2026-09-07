#!/usr/bin/env python3
"""3-D hull / tube plot across SINGER posterior sample embeddings.

For each individual, shows:
  - A semi-transparent tube whose cross-section at each time is the convex hull
    of that individual's 8 PS positions in X1/X2 space
  - A thick line for the posterior mean trajectory

Exports HTML (interactive) and PNG (static).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy.linalg import svd
from scipy.spatial import ConvexHull

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
    return aligned


def _resample_hull(xs, ys, n_angle):
    """
    Return n_angle evenly-arc-length-spaced points on the convex hull of (xs, ys).
    Falls back to a tiny circle when points are nearly collinear/coincident.
    """
    pts = np.column_stack([xs, ys])
    cx, cy = pts.mean(axis=0)

    try:
        if len(np.unique(pts, axis=0)) < 3:
            raise ValueError("degenerate")
        hull   = ConvexHull(pts)
        vx     = xs[hull.vertices]
        vy     = ys[hull.vertices]
        # Close the loop
        vx = np.append(vx, vx[0])
        vy = np.append(vy, vy[0])
        dx, dy   = np.diff(vx), np.diff(vy)
        cumlen   = np.concatenate([[0], np.cumsum(np.hypot(dx, dy))])
        total    = cumlen[-1]
        if total < 1e-12:
            raise ValueError("degenerate")
        targets  = np.linspace(0, total, n_angle, endpoint=False)
        rx = np.interp(targets, cumlen, vx)
        ry = np.interp(targets, cumlen, vy)
        return rx, ry
    except Exception:
        r     = max(np.std(xs), np.std(ys), 1e-6)
        theta = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
        return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _build_tube_mesh(rings_x, rings_y, rings_z, n_angle):
    """
    Build Mesh3d vertices and faces for a tube defined by stacked hull rings.
    rings_x/y: (n_time, n_angle) arrays
    rings_z:   (n_time,) array — same z for all vertices in a ring
    """
    n_time = rings_x.shape[0]
    # Vertices: (n_time * n_angle, 3)
    xs = rings_x.ravel()
    ys = rings_y.ravel()
    zs = np.repeat(rings_z, n_angle)

    i_faces, j_faces, k_faces = [], [], []
    for t in range(n_time - 1):
        for a in range(n_angle):
            a1 = (a + 1) % n_angle
            v0 = t       * n_angle + a
            v1 = t       * n_angle + a1
            v2 = (t + 1) * n_angle + a
            v3 = (t + 1) * n_angle + a1
            # Two triangles per quad
            i_faces += [v0, v1]
            j_faces += [v2, v2]
            k_faces += [v1, v3]

    return xs, ys, zs, np.array(i_faces), np.array(j_faces), np.array(k_faces)


def main():
    parser = argparse.ArgumentParser(
        description="Hull/tube plot across PS runs, one tube per individual"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_dir")
    group.add_argument("--inputs", nargs="+")

    parser.add_argument("--n_ps",          type=int,   default=8)
    parser.add_argument("--colors",        required=True)
    parser.add_argument("--labels",        required=True)
    parser.add_argument("--output_path",   required=True)
    parser.add_argument("--output_name",   default="embeddings_PS_hulls")
    parser.add_argument("--min_time",      type=float, default=1000)
    parser.add_argument("--max_time",      type=float, default=100000)
    parser.add_argument("--time_scale",    default="log")
    parser.add_argument("--lims",          type=str,   default="None")
    parser.add_argument("--n_angle",       type=int,   default=16,
                        help="Hull ring vertices per timepoint")
    parser.add_argument("--n_t_hull",      type=int,   default=60,
                        help="Timepoints sampled for hull (log-spaced)")
    parser.add_argument("--hull_opacity",  type=float, default=0.18)
    parser.add_argument("--mean_width",    type=float, default=3.0)
    parser.add_argument("--eye_x",         type=float, default=1.8)
    parser.add_argument("--eye_y",         type=float, default=1.8)
    parser.add_argument("--eye_z",         type=float, default=0.3)
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

    print("Procrustes-aligning...")
    dfs = _procrustes_align(dfs, id_col)

    # ── Compute mean df and build per-individual hull data ───────────────────
    combined = pd.concat(
        [df[[id_col, "time", "X1", "X2"]] for df in dfs], ignore_index=True
    )
    mean_df = combined.groupby([id_col, "time"], as_index=False)[["X1", "X2"]].mean()
    mean_df["population"] = mean_df[id_col].map(pop_map).fillna("unknown")

    # Common timepoints for hull rings (log-spaced)
    all_times = sorted(combined["time"].unique())
    if args.time_scale == "log":
        log_t = np.log10(np.array(all_times, dtype=float))
        idx   = np.round(np.linspace(0, len(all_times) - 1, args.n_t_hull)).astype(int)
    else:
        idx   = np.round(np.linspace(0, len(all_times) - 1, args.n_t_hull)).astype(int)
    hull_times = [all_times[i] for i in np.unique(idx)]

    if lims is None:
        all_x = combined["X1"].dropna().values
        all_y = combined["X2"].dropna().values
        cx    = (all_x.max() + all_x.min()) / 2
        cy    = (all_y.max() + all_y.min()) / 2
        half  = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.06
        xy_lims = max(abs(cx) + half, abs(cy) + half, half)
    else:
        xy_lims = lims

    # ── Initialise figure via PlotEmbeddings ────────────────────────────────
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

    def _tplot(t_arr):
        return np.maximum(np.asarray(t_arr, dtype=float), plotter.time_eps_plot)

    # ── Hull tubes — one Mesh3d per individual ───────────────────────────────
    print("Building hull tubes...")
    # Pre-pivot: for each individual, get a (n_ps, n_hull_times, 2) array
    # combined is already available; filter to hull_times
    hull_time_set = set(hull_times)
    hull_sub = combined[combined["time"].isin(hull_time_set)]

    populations_seen_hull = set()
    all_ids = sorted(dfs[0][id_col].unique())

    for ind_id in all_ids:
        pop   = pop_map.get(ind_id, "unknown")
        color = pop_color_map.get(pop, "#808080")

        sub = hull_sub[hull_sub[id_col] == ind_id]
        if sub.empty:
            continue

        # Group by time; at each time we have up to n_ps rows (one per PS)
        by_time = sub.groupby("time")

        rings_x = np.full((len(hull_times), args.n_angle), np.nan)
        rings_y = np.full((len(hull_times), args.n_angle), np.nan)

        for ti, t in enumerate(hull_times):
            if t not in by_time.groups:
                continue
            grp = by_time.get_group(t)
            xs  = grp["X1"].values
            ys  = grp["X2"].values
            rx, ry = _resample_hull(xs, ys, args.n_angle)
            rings_x[ti] = rx
            rings_y[ti] = ry

        # Drop timepoints with missing hull data
        valid = ~np.isnan(rings_x[:, 0])
        rings_x = rings_x[valid]
        rings_y = rings_y[valid]
        rings_z = _tplot(np.array(hull_times)[valid])

        if rings_x.shape[0] < 2:
            continue

        vx, vy, vz, fi, fj, fk = _build_tube_mesh(rings_x, rings_y, rings_z, args.n_angle)
        show_legend_hull = pop not in populations_seen_hull
        populations_seen_hull.add(pop)

        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=fi, j=fj, k=fk,
            color=color,
            opacity=args.hull_opacity,
            name=pop,
            hoverinfo="skip",
            showlegend=show_legend_hull,
            legendgroup=pop,
            flatshading=False,
        ))

    # ── Posterior mean trajectories ─────────────────────────────────────────
    print("Adding mean trajectories...")
    populations_seen_mean = set()
    for ind_id in all_ids:
        row = mean_df[mean_df[id_col] == ind_id].sort_values("time")
        if row.empty:
            continue
        pop   = pop_map.get(ind_id, "unknown")
        color = pop_color_map.get(pop, "#808080")
        show_legend = pop not in populations_seen_mean
        populations_seen_mean.add(pop)
        fig.add_trace(go.Scatter3d(
            x=row["X1"].values,
            y=row["X2"].values,
            z=_tplot(row["time"]),
            mode="lines",
            line=dict(color=color, width=args.mean_width),
            opacity=1.0,
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
