#!/usr/bin/env python3
"""
Overlay all chimps_lowres replicate embeddings on shared plots.

One replicate (--reference_ps, default 50) is drawn bold/opaque; all
others are drawn thin+transparent.  Procrustes-aligns every replicate to
the reference at the most-recent time slice.

Outputs (in --output_path):
  overlay_2d_slice.png      — 2-D scatter at most-recent time, all replicates overlaid
  overlay_3d_v1.png         — 3-D trajectories, view 1
  overlay_3d_v2.png         — 3-D trajectories, view 2
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.linalg import svd

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from PlotEmbeddings import PlotEmbeddings


# ── visual parameters ──────────────────────────────────────────────────────
BG_LW     = 0.15   # line-width for background replicates
BG_ALPHA  = 0.12   # opacity for background replicates
REF_LW    = 1.8    # line-width for reference replicate
REF_ALPHA = 0.95   # opacity for reference replicate
BG_S      = 5      # scatter dot size for background replicates
REF_S     = 22     # scatter dot size for reference replicate
LABEL_FS  = 7      # font size for individual labels on 2-D slice
AX_PAD    = 0.12   # fractional padding around reference-run bounding box


def _toned_color(hex_color, ps_idx, n_ps, lightness_range=0.18):
    """Return a lightness-shifted variant of hex_color for replicate ps_idx.

    Cycles evenly across [base - range/2, base + range/2] in HLS space so
    each replicate gets a subtly distinct tone of its population colour.
    """
    import colorsys
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # shift lightness with a sinusoidal sweep across replicates
    phase = ps_idx / max(n_ps - 1, 1)   # 0 … 1
    delta = lightness_range * (phase - 0.5)
    l_new = np.clip(l + delta, 0.15, 0.92)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s)
    return "#{:02x}{:02x}{:02x}".format(int(r2*255), int(g2*255), int(b2*255))


def _discover_embeddings(ps_dir):
    """Return path to the k=1 embeddings parquet in a PS output directory."""
    pattern = os.path.join(ps_dir, "data", "embeddings.*.k1.*.parquet")
    matches = [m for m in glob.glob(pattern) if "loss" not in os.path.basename(m)]
    if not matches:
        return None
    matches.sort(key=os.path.getmtime)
    return matches[-1]


def _procrustes_rotation(src_xy, ref_xy):
    """Return orthogonal rotation R (2×2) mapping src_xy → ref_xy (no scaling)."""
    src_mean = src_xy.mean(axis=0)
    ref_mean = ref_xy.mean(axis=0)
    U, _, Vt = svd((src_xy - src_mean).T @ (ref_xy - ref_mean))
    return U @ Vt, src_mean, ref_mean


def _apply_rotation(xy, R, src_mean, ref_mean):
    return (xy - src_mean) @ R + ref_mean


def main():
    parser = argparse.ArgumentParser(
        description="Overlay chimps_lowres replicate embeddings on shared plots"
    )
    parser.add_argument("--base_dir", required=True,
                        help="Directory containing PS0/, PS1/, … subdirs "
                             "(e.g. results/empirical/DeManuel/chimps_lowres)")
    parser.add_argument("--n_ps", type=int, default=100,
                        help="Number of posterior-sample replicates (default 100)")
    parser.add_argument("--reference_ps", type=int, default=50,
                        help="Index of the 'modal' replicate drawn bold (default 50)")
    parser.add_argument("--colors", required=True, help="Population colors CSV")
    parser.add_argument("--labels", required=True, help="Population labels CSV")
    parser.add_argument("--output_path", required=True,
                        help="Directory for output PNGs")
    parser.add_argument("--min_time", type=float, default=1000,
                        help="Minimum time for 3-D trajectory plots (gens)")
    parser.add_argument("--max_time", type=float, default=10000,
                        help="Maximum time for 3-D trajectory plots (gens)")
    parser.add_argument("--lims", type=str, default="None",
                        help="Shared XY axis half-range. 'None' = auto from data.")
    parser.add_argument("--time_scale", type=str, default="log",
                        choices=["log", "linear"],
                        help="Time axis scale for 3-D plot (default log)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    lims = None if args.lims == "None" else float(args.lims)

    # ── load colors / labels ──────────────────────────────────────────────
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)

    # ── discover & load replicates ────────────────────────────────────────
    print("Discovering embeddings files...")
    ps_indices = []
    raw_dfs    = []
    for k in range(args.n_ps):
        ps_dir = os.path.join(args.base_dir, f"PS{k}")
        path   = _discover_embeddings(ps_dir)
        if path is None:
            print(f"  PS{k}: no k=1 embeddings found, skipping")
            continue
        df = pd.read_parquet(path)
        ps_indices.append(k)
        raw_dfs.append(df)
        print(f"  PS{k}: {os.path.basename(path)}  ({len(df)} rows)")

    if args.reference_ps not in ps_indices:
        raise ValueError(
            f"Reference PS{args.reference_ps} not found among available replicates: {ps_indices}"
        )

    ref_idx = ps_indices.index(args.reference_ps)

    # ── id column + labels ────────────────────────────────────────────────
    id_col = "individual_id" if "individual_id" in raw_dfs[0].columns else "sample_id"
    if id_col == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    labels_df = labels_df.drop_duplicates(subset=id_col)

    active_ids  = raw_dfs[ref_idx][id_col].unique()
    labels_filt = labels_df[labels_df[id_col].isin(active_ids)]
    pop_map     = labels_filt.set_index(id_col)["population"]
    populations = labels_filt["population"].unique()

    colors_filt  = colors_df[colors_df["population"].isin(populations)]
    color_map    = colors_filt.set_index("population")["color"].to_dict()

    # ── Procrustes alignment: align all PS to reference at most-recent slice ──
    # Use min_time-filtered data for the alignment reference
    print("\nAligning replicates to PS{} (Procrustes)...".format(args.reference_ps))

    dfs_filtered = []
    for df in raw_dfs:
        dfs_filtered.append(df[(df["time"] >= args.min_time) & (df["time"] <= args.max_time)])

    # Most recent common time across filtered replicates
    most_recent = min(df["time"].min() for df in dfs_filtered)
    common_ids  = sorted(
        set.intersection(*[set(df[df["time"] == most_recent][id_col].values) for df in dfs_filtered])
    )

    ref_slice = dfs_filtered[ref_idx][dfs_filtered[ref_idx]["time"] == most_recent].set_index(id_col)
    ref_xy    = ref_slice.loc[common_ids, ["X1", "X2"]].values

    aligned = []
    for i, (df, k) in enumerate(zip(dfs_filtered, ps_indices)):
        if i == ref_idx:
            aligned.append(df.copy())
            continue
        sl    = df[df["time"] == most_recent].set_index(id_col)
        src   = sl.loc[common_ids, ["X1", "X2"]].values
        R, sm, rm = _procrustes_rotation(src, ref_xy)
        all_xy = df[["X1", "X2"]].values
        rot    = _apply_rotation(all_xy, R, sm, rm)
        d2     = df.copy()
        d2["X1"] = rot[:, 0]
        d2["X2"] = rot[:, 1]
        aligned.append(d2)

    print(f"Aligned {len(aligned)} replicates.  Most-recent time = {most_recent:.1f} gen")

    # ── shared axis limits — fit to reference run only ────────────────────
    if lims is None:
        ref_df   = aligned[ref_idx]
        ref_x    = ref_df["X1"].dropna().values
        ref_y    = ref_df["X2"].dropna().values
        combined = np.concatenate([ref_x, ref_y])
        span     = combined.max() - combined.min()
        pad      = AX_PAD * span
        lims     = max(abs(combined.min() - pad), abs(combined.max() + pad))
        print(f"Auto XY limits (from PS{args.reference_ps}): ±{lims:.2f}")

    # ── 2-D slice plot ─────────────────────────────────────────────────────
    print("\nBuilding 2-D slice overlay...")
    fig2d, ax2d = plt.subplots(figsize=(6, 6))

    n_ps_loaded = len(ps_indices)
    for i, (df, k) in enumerate(zip(aligned, ps_indices)):
        is_ref  = (k == args.reference_ps)
        sl      = df[df["time"] == most_recent].copy()
        # per-PS tone: shift each population's base colour slightly
        def _row_color(ind_id):
            pop  = pop_map.get(ind_id, None)
            base = color_map.get(pop, "#808080") if pop else "#808080"
            return base if is_ref else _toned_color(base, i, n_ps_loaded)
        sl["color"] = sl[id_col].apply(_row_color)

        ax2d.scatter(
            sl["X1"], sl["X2"],
            c=sl["color"],
            s=REF_S if is_ref else BG_S,
            alpha=REF_ALPHA if is_ref else BG_ALPHA,
            linewidths=0,
            zorder=10 if is_ref else 2,
        )

    # Labels on reference dots
    ref_sl = aligned[ref_idx][aligned[ref_idx]["time"] == most_recent].copy()
    ref_sl["color"] = ref_sl[id_col].map(pop_map).map(color_map).fillna("#808080")
    label_offset = 0.015 * 2 * lims
    for _, row in ref_sl.iterrows():
        pop = pop_map.get(row[id_col], "?")
        ax2d.text(row["X1"], row["X2"] - label_offset,
                  f"{pop[:2]}-{row[id_col]}",
                  fontsize=LABEL_FS, ha="center", va="top",
                  color=color_map.get(pop, "#808080"), fontweight="bold",
                  zorder=20)

    ax2d.set_xlim(-lims, lims)
    ax2d.set_ylim(-lims, lims)
    ax2d.set_aspect("equal")
    ax2d.set_xlabel("X1")
    ax2d.set_ylabel("X2")
    ax2d.set_title(
        f"Chimps lowres — 2-D slice at t = {most_recent:.0f} gen\n"
        f"n = {len(ps_indices)} replicates, PS{args.reference_ps} bold",
        fontsize=9,
    )

    legend_handles = [
        mpatches.Patch(facecolor=color_map[pop], label=pop)
        for pop in populations if pop in color_map
    ]
    ax2d.legend(handles=legend_handles, title="Population",
                bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig2d.tight_layout()
    out2d = os.path.join(args.output_path, "overlay_2d_slice.png")
    fig2d.savefig(out2d, dpi=300, bbox_inches="tight")
    plt.close(fig2d)
    print(f"Saved {out2d}")

    # ── 2-D density overlay ────────────────────────────────────────────────
    # Pool X1/X2 at most-recent slice across ALL replicates, then draw a
    # filled KDE contour per population.  Reference scatter on top.
    print("Building 2-D density overlay...")
    from scipy.stats import gaussian_kde

    fig_kde, ax_kde = plt.subplots(figsize=(6, 6))

    # Pool positions per population across all replicates
    pop_xy: dict = {pop: {"x": [], "y": []} for pop in populations}
    for df in aligned:
        sl = df[df["time"] == most_recent].copy()
        for ind_id in sl[id_col].values:
            pop = pop_map.get(ind_id, None)
            if pop is None or pop not in pop_xy:
                continue
            row = sl[sl[id_col] == ind_id]
            pop_xy[pop]["x"].append(float(row["X1"].values[0]))
            pop_xy[pop]["y"].append(float(row["X2"].values[0]))

    grid_n = 200
    xg = np.linspace(-lims, lims, grid_n)
    yg = np.linspace(-lims, lims, grid_n)
    Xg, Yg = np.meshgrid(xg, yg)
    pts = np.vstack([Xg.ravel(), Yg.ravel()])

    for pop in populations:
        xs = np.array(pop_xy[pop]["x"])
        ys = np.array(pop_xy[pop]["y"])
        if len(xs) < 3:
            continue
        base_col = color_map.get(pop, "#808080")
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
            Z = kde(pts).reshape(grid_n, grid_n)
            # Levels from sparse outer (1%) to dense inner (99%) of peak.
            # Sentinel at 100.1% ensures the innermost band is filled by contourf.
            thresholds = [0.01, 0.20, 0.50, 0.80, 0.99]
            levels      = [Z.max() * t for t in thresholds]
            fill_levels = levels + [Z.max() * 1.001]
            ax_kde.contourf(Xg, Yg, Z, levels=fill_levels,
                            colors=[base_col], alpha=0.18)
            ax_kde.contour(Xg, Yg, Z, levels=levels,
                           colors=[base_col], linewidths=0.7, alpha=0.6)
        except Exception:
            pass  # too few unique points for KDE

    # Reference scatter on top
    ref_sl2 = aligned[ref_idx][aligned[ref_idx]["time"] == most_recent].copy()
    ref_sl2["color"] = ref_sl2[id_col].map(pop_map).map(color_map).fillna("#808080")
    ax_kde.scatter(ref_sl2["X1"], ref_sl2["X2"],
                   c=ref_sl2["color"], s=REF_S, alpha=REF_ALPHA,
                   linewidths=0.4, edgecolors="white", zorder=10)
    label_offset = 0.015 * 2 * lims
    for _, row in ref_sl2.iterrows():
        pop = pop_map.get(row[id_col], "?")
        ax_kde.text(row["X1"], row["X2"] - label_offset,
                    f"{pop[:2]}-{row[id_col]}",
                    fontsize=LABEL_FS, ha="center", va="top",
                    color=color_map.get(pop, "#808080"), fontweight="bold",
                    zorder=20)

    ax_kde.set_xlim(-lims, lims)
    ax_kde.set_ylim(-lims, lims)
    ax_kde.set_aspect("equal")
    ax_kde.set_xlabel("X1")
    ax_kde.set_ylabel("X2")
    ax_kde.set_title(
        f"Chimps lowres — density at t = {most_recent:.0f} gen\n"
        f"n = {len(ps_indices)} replicates, PS{args.reference_ps} scatter",
        fontsize=9,
    )
    ax_kde.legend(handles=legend_handles, title="Population",
                  bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    fig_kde.tight_layout()
    out_kde = os.path.join(args.output_path, "overlay_2d_density.png")
    fig_kde.savefig(out_kde, dpi=300, bbox_inches="tight")
    plt.close(fig_kde)
    print(f"Saved {out_kde}")

    # ── 3-D trajectory overlay (Plotly, matching normal embedding plots) ──────
    print("\nBuilding 3-D trajectory overlay (Plotly)...")
    import plotly.graph_objects as go
    import plotly.io as pio

    # Initialise via PlotEmbeddings on the reference replicate so axis style
    # and tick configuration match every other embeddings plot in the pipeline.
    plotter = PlotEmbeddings(
        embeddings_df=aligned[ref_idx],
        labels_df=labels_filt,
        colors_df=colors_filt,
        sample_id_col=id_col,
    )
    plotter.init_plot(
        time_scale=args.time_scale,
        xy_lims=lims,
        min_time=args.min_time,
        max_time=args.max_time,
    )

    # PlotEmbeddings stores the plot-safe time column name
    tplot_col = plotter.tplot_col  # "time__plot"
    time_eps  = plotter.time_eps_plot

    def _tplot(t_arr):
        """Mirror PlotEmbeddings' log-safe time transform."""
        if args.time_scale == "log":
            return np.maximum(t_arr, time_eps)
        return t_arr

    # Add background replicates as thin Scatter3d traces (inserted before ref)
    n_ps_loaded = len(ps_indices)
    bg_traces = []
    for i, (df, k) in enumerate(zip(aligned, ps_indices)):
        if k == args.reference_ps:
            continue
        for ind_id in sorted(df[id_col].unique()):
            traj = df[df[id_col] == ind_id].sort_values("time")
            pop  = pop_map.get(ind_id, None)
            base = color_map.get(pop, "#808080") if pop else "#808080"
            col  = _toned_color(base, i, n_ps_loaded)
            zvals = _tplot(traj["time"].values)
            bg_traces.append(go.Scatter3d(
                x=traj["X1"].values,
                y=traj["X2"].values,
                z=zvals,
                mode="lines",
                line=dict(color=col, width=BG_LW * 4),  # Plotly widths ~4× matplotlib
                opacity=BG_ALPHA,
                showlegend=False,
                hoverinfo="skip",
            ))

    # Shuffle so no population is systematically rendered on top of another
    rng = np.random.default_rng(seed=0)
    rng.shuffle(bg_traces)
    plotter.fig.add_traces(bg_traces)

    # Reference replicate trajectories on top (bold, full population colours)
    plotter.add_sample_trajectories(line_width=REF_LW * 2, alpha=REF_ALPHA)
    plotter.add_legend()

    plotter.fig.update_layout(
        title=dict(
            text=(f"Chimps lowres — {n_ps_loaded} replicates, "
                  f"PS{args.reference_ps} bold"),
            font=dict(size=12),
        )
    )

    # HTML (fully interactive)
    out_html = os.path.join(args.output_path, "overlay_3d.html")
    plotter.fig.write_html(out_html)
    print(f"Saved {out_html}")

    # Static PNGs — two camera angles matching the standard pipeline views
    out_v1 = os.path.join(args.output_path, "overlay_3d_v1")
    img_v1, _ = plotter.save_static(
        width=600, height=600, scale=2,
        eye=dict(x=1.8, y=1.8, z=0.3),
        up=dict(x=0, y=0, z=1),
    )
    img_v1.save(f"{out_v1}.png")
    print(f"Saved {out_v1}.png")

    out_v2 = os.path.join(args.output_path, "overlay_3d_v2")
    img_v2, _ = plotter.save_static(
        width=600, height=600, scale=2,
        eye=dict(x=-0.75, y=-1.5, z=0.3),
        up=dict(x=0, y=0, z=1),
    )
    img_v2.save(f"{out_v2}.png")
    print(f"Saved {out_v2}.png")

    # v3: top-down (X1/X2 plane, time axis pointing away)
    out_v3 = os.path.join(args.output_path, "overlay_3d_v3")
    img_v3, _ = plotter.save_static(
        width=600, height=600, scale=2,
        eye=dict(x=0.0, y=0.0, z=2.5),
        up=dict(x=0, y=1, z=0),
    )
    img_v3.save(f"{out_v3}.png")
    print(f"Saved {out_v3}.png")

    # v4: side-on along X2 axis (shows time vs X1 profile)
    out_v4 = os.path.join(args.output_path, "overlay_3d_v4")
    img_v4, _ = plotter.save_static(
        width=600, height=600, scale=2,
        eye=dict(x=2.5, y=0.0, z=0.3),
        up=dict(x=0, y=0, z=1),
    )
    img_v4.save(f"{out_v4}.png")
    print(f"Saved {out_v4}.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
